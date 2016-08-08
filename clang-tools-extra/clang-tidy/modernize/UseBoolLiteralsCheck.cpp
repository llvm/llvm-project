//===--- UseBoolLiteralsCheck.cpp - clang-tidy-----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "UseBoolLiteralsCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace modernize {

void UseBoolLiteralsCheck::registerMatchers(MatchFinder *Finder) {
  if (!getLangOpts().CPlusPlus)
    return;

  Finder->addMatcher(
      implicitCastExpr(
          has(ignoringParenImpCasts(integerLiteral().bind("literal"))),
          hasImplicitDestinationType(qualType(booleanType())),
          unless(isInTemplateInstantiation()),
          anyOf(hasParent(explicitCastExpr().bind("cast")), anything())),
      this);

  Finder->addMatcher(
      conditionalOperator(
          hasParent(implicitCastExpr(
              hasImplicitDestinationType(qualType(booleanType())),
              unless(isInTemplateInstantiation()))),
          eachOf(hasTrueExpression(
                     ignoringParenImpCasts(integerLiteral().bind("literal"))),
                 hasFalseExpression(
                     ignoringParenImpCasts(integerLiteral().bind("literal"))))),
      this);
}

void UseBoolLiteralsCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Literal = Result.Nodes.getNodeAs<IntegerLiteral>("literal");
  const auto *Cast = Result.Nodes.getNodeAs<Expr>("cast");
  bool LiteralBooleanValue = Literal->getValue().getBoolValue();

  if (Literal->isInstantiationDependent())
    return;

  const Expr *Expression = Cast ? Cast : Literal;

  auto Diag =
      diag(Expression->getExprLoc(),
           "converting integer literal to bool, use bool literal instead");

  if (!Expression->getLocStart().isMacroID())
    Diag << FixItHint::CreateReplacement(
        Expression->getSourceRange(), LiteralBooleanValue ? "true" : "false");
}

} // namespace modernize
} // namespace tidy
} // namespace clang
