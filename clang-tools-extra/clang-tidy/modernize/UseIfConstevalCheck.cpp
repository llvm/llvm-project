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
#include "clang/Basic/CharInfo.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::modernize {

namespace {

struct BraceFix {
  bool NeedsBraces = false;
  utils::BraceInsertionHints Hints;
};

} // namespace

static std::optional<SourceRange> getHeaderRange(const IfStmt *If,
                                                 const SourceManager &SM,
                                                 const LangOptions &LangOpts) {
  if (If->getLParenLoc().isMacroID() || If->getRParenLoc().isMacroID())
    return std::nullopt;

  const SourceRange HeaderRange(If->getLParenLoc(), If->getRParenLoc());
  // Validate that the token range is safely rewriteable in file source before
  // offering a fix-it.
  if (Lexer::makeFileCharRange(CharSourceRange::getTokenRange(HeaderRange), SM,
                               LangOpts)
          .isInvalid())
    return std::nullopt;
  return HeaderRange;
}

static std::optional<BraceFix>
getBraceFix(const Stmt *S, const LangOptions &LangOpts, const SourceManager &SM,
            SourceLocation StartLoc,
            SourceLocation EndLocHint = SourceLocation()) {
  if (S)
    S = S->stripLabelLikeStatements();
  if (!S || isa<CompoundStmt>(S))
    return BraceFix();

  const auto Hints =
      utils::getBraceInsertionsHints(S, LangOpts, SM, StartLoc, EndLocHint);
  if (!Hints || !Hints.offersFixIts())
    return std::nullopt;

  return BraceFix{true, Hints};
}

static bool needsLeadingSpaceBeforeConsteval(SourceLocation LParenLoc,
                                             const SourceManager &SM) {
  bool Invalid = false;
  const char *LParen = SM.getCharacterData(LParenLoc, &Invalid);
  return Invalid || !isWhitespace(LParen[-1]);
}

void UseIfConstevalCheck::registerMatchers(MatchFinder *Finder) {
  const auto IsConstantEvaluatedCall =
      callExpr(callee(functionDecl(hasName("is_constant_evaluated"),
                                   isInStdNamespace())))
          .bind("call");
  const auto IsNegatedConstantEvaluatedExpr =
      unaryOperator(hasOperatorName("!"),
                    hasUnaryOperand(ignoringParens(IsConstantEvaluatedCall)))
          .bind("negation");
  const auto IsConstantEvaluatedExpr = ignoringParenImpCasts(
      anyOf(IsConstantEvaluatedCall, IsNegatedConstantEvaluatedExpr));

  Finder->addMatcher(
      ifStmt(unless(isConstexpr()),
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
  const llvm::StringRef ConstevalClause =
      IsNegated ? "!consteval" : "consteval";
  const SourceLocation DiagLoc =
      Result.SourceManager->getExpansionLoc(Call->getExprLoc());

  auto Diag = diag(DiagLoc, "use 'if %0' instead of checking "
                            "'std::is_constant_evaluated()'")
              << ConstevalClause;

  if (If->hasInitStorage() || If->hasVarStorage())
    return;

  std::optional<SourceRange> HeaderRange =
      getHeaderRange(If, *Result.SourceManager, getLangOpts());
  if (!HeaderRange)
    return;

  std::optional<BraceFix> ThenBraceFix =
      getBraceFix(If->getThen(), getLangOpts(), *Result.SourceManager,
                  If->getRParenLoc(), If->getElseLoc());
  if (!ThenBraceFix)
    return;

  std::optional<BraceFix> ElseBraceFix = BraceFix();
  if (If->getElse()) {
    ElseBraceFix = getBraceFix(If->getElse(), getLangOpts(),
                               *Result.SourceManager, If->getElseLoc());
  }
  if (!ElseBraceFix)
    return;

  const bool NeedsLeadingSpace = needsLeadingSpaceBeforeConsteval(
      If->getLParenLoc(), *Result.SourceManager);
  const std::string HeaderReplacement = [&] {
    std::string Replacement = ConstevalClause.str();
    if (NeedsLeadingSpace)
      Replacement.insert(0, 1, ' ');
    if (ThenBraceFix->NeedsBraces)
      Replacement += " {";
    return Replacement;
  }();
  Diag << FixItHint::CreateReplacement(*HeaderRange, HeaderReplacement);

  if (ThenBraceFix->NeedsBraces)
    Diag << ThenBraceFix->Hints.closingBraceFixIt();

  if (ElseBraceFix && ElseBraceFix->NeedsBraces)
    Diag << ElseBraceFix->Hints.openingBraceFixIt()
         << ElseBraceFix->Hints.closingBraceFixIt();
}

} // namespace clang::tidy::modernize
