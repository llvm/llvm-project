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
        };
    return {Mapping};
  }
};

} // namespace clang::tidy

namespace clang::tidy::readability {

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
      CommaPolicy(Options.get("CommaPolicy", CommaPolicyKind::Append)),
      EnumThreshold(Options.get("EnumThreshold", 1U)),
      InitListThreshold(Options.get("InitListThreshold", 3U)) {}

void TrailingCommaCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "CommaPolicy", CommaPolicy);
  Options.store(Opts, "EnumThreshold", EnumThreshold);
  Options.store(Opts, "InitListThreshold", InitListThreshold);
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
  // Count enumerators and get the last one
  unsigned NumEnumerators = 0;
  const EnumConstantDecl *LastEnumerator = nullptr;
  for (const EnumConstantDecl *ECD : Enum->enumerators()) {
    LastEnumerator = ECD;
    ++NumEnumerators;
  }
  assert(LastEnumerator);
  assert(NumEnumerators > 0);

  if (NumEnumerators < EnumThreshold)
    return;

  SourceLocation LastEnumLoc;
  if (const Expr *Init = LastEnumerator->getInitExpr())
    LastEnumLoc = Init->getEndLoc();
  else
    LastEnumLoc = LastEnumerator->getLocation();

  if (LastEnumLoc.isInvalid())
    return;

  emitDiag(LastEnumLoc, DiagKind::Enum, Result);
}

void TrailingCommaCheck::checkInitListExpr(
    const InitListExpr *InitList, const MatchFinder::MatchResult &Result) {
  // We need to use the syntactic form for correct source locations.
  if (InitList->isSemanticForm())
    if (const InitListExpr *SyntacticForm = InitList->getSyntacticForm())
      InitList = SyntacticForm;

  const unsigned NumInits = InitList->getNumInits();
  if (NumInits < InitListThreshold)
    return;

  const Expr *LastInit = InitList->getInit(NumInits - 1);
  assert(LastInit);

  // Skip pack expansions - they already have special syntax with '...'
  if (isa<PackExpansionExpr>(LastInit))
    return;

  const SourceLocation LastInitLoc = LastInit->getEndLoc();
  if (LastInitLoc.isInvalid())
    return;

  emitDiag(LastInitLoc, DiagKind::InitList, Result);
}

void TrailingCommaCheck::emitDiag(
    SourceLocation LastLoc, DiagKind Kind,
    const ast_matchers::MatchFinder::MatchResult &Result) {
  const std::optional<Token> NextToken =
      utils::lexer::findNextTokenSkippingComments(
          LastLoc, *Result.SourceManager, getLangOpts());
  if (!NextToken)
    return;

  const bool HasTrailingComma = NextToken->is(tok::comma);
  const SourceLocation InsertLoc = Lexer::getLocForEndOfToken(
      LastLoc, 0, *Result.SourceManager, getLangOpts());

  if (CommaPolicy == CommaPolicyKind::Append && !HasTrailingComma) {
    diag(InsertLoc, "%select{initializer list|enum}0 should have "
                    "a trailing comma")
        << Kind << FixItHint::CreateInsertion(InsertLoc, ",");
  } else if (CommaPolicy == CommaPolicyKind::Remove && HasTrailingComma) {
    const SourceLocation CommaLoc = NextToken->getLocation();
    if (CommaLoc.isInvalid())
      return;
    diag(CommaLoc, "%select{initializer list|enum}0 should not have "
                   "a trailing comma")
        << Kind << FixItHint::CreateRemoval(CommaLoc);
  }
}

} // namespace clang::tidy::readability
