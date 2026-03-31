//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseIfConstevalCheck.h"

#include "../utils/BracesAroundStatement.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::modernize {

namespace {

struct BraceFix {
  bool NeedsBraces = false;
  utils::BraceInsertionHints Hints;
};

} // namespace

static const Stmt *ignoreAttributedStmt(const Stmt *S) {
  while (const auto *AS = dyn_cast_or_null<AttributedStmt>(S))
    S = AS->getSubStmt();
  return S;
}

static std::optional<CharSourceRange>
getHeaderRange(const IfStmt *If, const SourceManager &SM,
               const LangOptions &LangOpts) {
  if (If->getIfLoc().isMacroID() || If->getRParenLoc().isMacroID())
    return std::nullopt;

  const CharSourceRange HeaderRange = Lexer::makeFileCharRange(
      CharSourceRange::getTokenRange(If->getIfLoc(), If->getRParenLoc()), SM,
      LangOpts);
  if (HeaderRange.isInvalid())
    return std::nullopt;
  return HeaderRange;
}

static std::optional<BraceFix>
getBraceFix(const Stmt *S, const LangOptions &LangOpts, const SourceManager &SM,
            SourceLocation StartLoc,
            SourceLocation EndLocHint = SourceLocation()) {
  S = ignoreAttributedStmt(S);
  if (!S || isa<CompoundStmt>(S))
    return BraceFix();

  const auto Hints =
      utils::getBraceInsertionsHints(S, LangOpts, SM, StartLoc, EndLocHint);
  if (!Hints || !Hints.offersFixIts())
    return std::nullopt;

  return BraceFix{true, Hints};
}
void UseIfConstevalCheck::registerMatchers(MatchFinder *Finder) {
  const auto IsConstantEvaluatedCall =
      callExpr(callee(functionDecl(hasName("is_constant_evaluated"),
                                   isInStdNamespace())))
          .bind("call");
  const auto IsNegatedConstantEvaluatedExpr =
      unaryOperator(hasOperatorName("!"), hasUnaryOperand(ignoringParenImpCasts(
                                              IsConstantEvaluatedCall)))
          .bind("negation");
  const auto IsConstantEvaluatedExpr =
      expr(anyOf(ignoringParenImpCasts(IsConstantEvaluatedCall),
                 ignoringParenImpCasts(IsNegatedConstantEvaluatedExpr)));

  Finder->addMatcher(
      ifStmt(unless(anyOf(isConstexpr(), isConsteval())),
             anyOf(hasCondition(IsConstantEvaluatedExpr),
                   hasConditionVariableStatement(declStmt(hasSingleDecl(
                       varDecl(hasInitializer(IsConstantEvaluatedExpr)))))))
          .bind("if"),
      this);
}

void UseIfConstevalCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *If = Result.Nodes.getNodeAs<IfStmt>("if");
  const auto *Call = Result.Nodes.getNodeAs<CallExpr>("call");
  assert(If && Call && "expected to match an if statement and its call");

  const bool IsNegated = Result.Nodes.getNodeAs<UnaryOperator>("negation");
  const char *Replacement = IsNegated ? "if !consteval" : "if consteval";
  const SourceLocation DiagLoc =
      Result.SourceManager->getExpansionLoc(Call->getExprLoc());

  auto Diag =
      diag(
          DiagLoc,
          "use '%0' instead of checking 'std::is_constant_evaluated()' in this "
          "'if' statement")
      << Replacement;

  if (If->hasInitStorage() || If->hasVarStorage())
    return;

  std::optional<CharSourceRange> HeaderRange =
      getHeaderRange(If, *Result.SourceManager, getLangOpts());
  if (!HeaderRange)
    return;

  std::optional<BraceFix> ThenBraceFix =
      getBraceFix(If->getThen(), getLangOpts(), *Result.SourceManager,
                  If->getRParenLoc(), If->getElseLoc());
  if (!ThenBraceFix)
    return;

  std::optional<BraceFix> ElseBraceFix = BraceFix();
  const Stmt *Else = ignoreAttributedStmt(If->getElse());
  if (Else && !isa<IfStmt>(Else)) {
    ElseBraceFix = getBraceFix(If->getElse(), getLangOpts(),
                               *Result.SourceManager, If->getElseLoc());
  }

  std::string HeaderReplacement(Replacement);
  if (ThenBraceFix->NeedsBraces)
    HeaderReplacement += " {";
  Diag << FixItHint::CreateReplacement(*HeaderRange, HeaderReplacement);

  if (ThenBraceFix->NeedsBraces)
    Diag << ThenBraceFix->Hints.closingBraceFixIt();

  if (ElseBraceFix && ElseBraceFix->NeedsBraces)
    Diag << ElseBraceFix->Hints.openingBraceFixIt()
         << ElseBraceFix->Hints.closingBraceFixIt();
}

} // namespace clang::tidy::modernize
