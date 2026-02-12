//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TrailingCommaCheck.h"
#include "../utils/LexerUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy {

template <>
struct OptionEnumMapping<readability::TrailingCommaCheck::CommaPolicyKind> {
  static llvm::ArrayRef<
      std::pair<readability::TrailingCommaCheck::CommaPolicyKind, StringRef>>
  getEnumMapping() {
    static constexpr std::pair<readability::TrailingCommaCheck::CommaPolicyKind,
                               StringRef>
        Mapping[] = {
            {readability::TrailingCommaCheck::CommaPolicyKind::Append,
             "Append"},
            {readability::TrailingCommaCheck::CommaPolicyKind::Remove,
             "Remove"},
            {readability::TrailingCommaCheck::CommaPolicyKind::Ignore,
             "Ignore"},
        };
    return {Mapping};
  }
};

} // namespace clang::tidy

namespace clang::tidy::readability {

static bool isSingleLine(SourceRange Range, const SourceManager &SM) {
  return SM.getExpansionLineNumber(Range.getBegin()) ==
         SM.getExpansionLineNumber(Range.getEnd());
}

namespace {

AST_POLYMORPHIC_MATCHER(isMacro,
                        AST_POLYMORPHIC_SUPPORTED_TYPES(EnumDecl,
                                                        InitListExpr)) {
  return Node.getBeginLoc().isMacroID() || Node.getEndLoc().isMacroID();
}

AST_MATCHER(EnumDecl, isEmptyEnum) { return Node.enumerators().empty(); }

AST_MATCHER(InitListExpr, isEmptyInitList) { return Node.getNumInits() == 0; }

} // namespace

TrailingCommaCheck::TrailingCommaCheck(StringRef Name,
                                       ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      SingleLineCommaPolicy(
          Options.get("SingleLineCommaPolicy", CommaPolicyKind::Remove)),
      MultiLineCommaPolicy(
          Options.get("MultiLineCommaPolicy", CommaPolicyKind::Append)) {
  if (SingleLineCommaPolicy == CommaPolicyKind::Ignore &&
      MultiLineCommaPolicy == CommaPolicyKind::Ignore)
    configurationDiag("The check '%0' will not perform any analysis because "
                      "'SingleLineCommaPolicy' and 'MultiLineCommaPolicy' are "
                      "both set to 'Ignore'.")
        << Name;
}

void TrailingCommaCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "SingleLineCommaPolicy", SingleLineCommaPolicy);
  Options.store(Opts, "MultiLineCommaPolicy", MultiLineCommaPolicy);
}

void TrailingCommaCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      enumDecl(isDefinition(), unless(isEmptyEnum()), unless(isMacro()))
          .bind("enum"),
      this);

  Finder->addMatcher(initListExpr(unless(isEmptyInitList()), unless(isMacro()))
                         .bind("initlist"),
                     this);
}

void TrailingCommaCheck::check(const MatchFinder::MatchResult &Result) {
  if (const auto *Enum = Result.Nodes.getNodeAs<EnumDecl>("enum"))
    checkEnumDecl(Enum, Result);
  else if (const auto *InitList =
               Result.Nodes.getNodeAs<InitListExpr>("initlist"))
    checkInitListExpr(InitList, Result);
  else
    llvm_unreachable("No matches found");
}

void TrailingCommaCheck::checkEnumDecl(const EnumDecl *Enum,
                                       const MatchFinder::MatchResult &Result) {
  const bool IsSingleLine = isSingleLine(
      {Enum->getBeginLoc(), Enum->getEndLoc()}, *Result.SourceManager);
  const CommaPolicyKind Policy =
      IsSingleLine ? SingleLineCommaPolicy : MultiLineCommaPolicy;

  if (Policy == CommaPolicyKind::Ignore)
    return;

  const std::optional<Token> LastTok =
      Lexer::findPreviousToken(Enum->getBraceRange().getEnd(),
                               *Result.SourceManager, getLangOpts(), false);
  if (!LastTok)
    return;

  emitDiag(LastTok->getLocation(), LastTok, DiagKind::Enum, Result, Policy);
}

void TrailingCommaCheck::checkInitListExpr(
    const InitListExpr *InitList, const MatchFinder::MatchResult &Result) {
  // We need to use non-empty syntactic form for correct source locations.
  if (const InitListExpr *SynInitInitList = InitList->getSyntacticForm();
      SynInitInitList && SynInitInitList->getNumInits() > 0)
    InitList = SynInitInitList;

  const bool IsSingleLine = isSingleLine(
      {InitList->getBeginLoc(), InitList->getEndLoc()}, *Result.SourceManager);
  const CommaPolicyKind Policy =
      IsSingleLine ? SingleLineCommaPolicy : MultiLineCommaPolicy;

  if (Policy == CommaPolicyKind::Ignore)
    return;

  const Expr *LastInit = InitList->inits().back();
  assert(LastInit);

  // Skip pack expansions - they already have special syntax with '...'
  if (isa<PackExpansionExpr>(LastInit))
    return;

  const std::optional<Token> NextTok =
      utils::lexer::findNextTokenSkippingComments(
          LastInit->getEndLoc(), *Result.SourceManager, getLangOpts());

  // If the next token is neither a comma nor closing brace, there might be
  // a macro (e.g., #define COMMA ,) that we can't safely analyze.
  if (NextTok && !NextTok->isOneOf(tok::comma, tok::r_brace))
    return;

  emitDiag(LastInit->getEndLoc(), NextTok, DiagKind::InitList, Result, Policy);
}

void TrailingCommaCheck::emitDiag(
    SourceLocation LastLoc, std::optional<Token> Token, DiagKind Kind,
    const ast_matchers::MatchFinder::MatchResult &Result,
    CommaPolicyKind Policy) {
  if (LastLoc.isInvalid() || !Token)
    return;

  const bool HasTrailingComma = Token->is(tok::comma);
  if (Policy == CommaPolicyKind::Append && !HasTrailingComma) {
    const SourceLocation InsertLoc = Lexer::getLocForEndOfToken(
        LastLoc, 0, *Result.SourceManager, getLangOpts());
    diag(InsertLoc, "%select{initializer list|enum}0 should have "
                    "a trailing comma")
        << Kind << FixItHint::CreateInsertion(InsertLoc, ",");
  } else if (Policy == CommaPolicyKind::Remove && HasTrailingComma) {
    const SourceLocation CommaLoc = Token->getLocation();
    if (CommaLoc.isInvalid())
      return;
    diag(CommaLoc, "%select{initializer list|enum}0 should not have "
                   "a trailing comma")
        << Kind << FixItHint::CreateRemoval(CommaLoc);
  }
}

} // namespace clang::tidy::readability
