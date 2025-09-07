//===--- UseBoolLiteralsCheck.cpp - clang-tidy-----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseBoolLiteralsCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::modernize {

UseBoolLiteralsCheck::UseBoolLiteralsCheck(StringRef Name,
                                           ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      IgnoreMacros(Options.get("IgnoreMacros", true)) {}

void UseBoolLiteralsCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "IgnoreMacros", IgnoreMacros);
}

void UseBoolLiteralsCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      traverse(TK_AsIs,
               implicitCastExpr(
                   has(ignoringParenImpCasts(integerLiteral().bind("literal"))),
                   hasImplicitDestinationType(qualType(booleanType())),
                   unless(isInTemplateInstantiation()),
                   optionally(hasParent(explicitCastExpr().bind("cast"))))),
      this);

  Finder->addMatcher(
      traverse(TK_AsIs,
               conditionalOperator(
                   hasParent(implicitCastExpr(
                       hasImplicitDestinationType(qualType(booleanType())),
                       unless(isInTemplateInstantiation()))),
                   eachOf(hasTrueExpression(ignoringParenImpCasts(
                              integerLiteral().bind("literal"))),
                          hasFalseExpression(ignoringParenImpCasts(
                              integerLiteral().bind("literal")))))),
      this);
}

void UseBoolLiteralsCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Literal = Result.Nodes.getNodeAs<IntegerLiteral>("literal");
  const auto *Cast = Result.Nodes.getNodeAs<Expr>("cast");
  bool LiteralBooleanValue = Literal->getValue().getBoolValue();

  if (Literal->isInstantiationDependent())
    return;

  const Expr *Expression = Cast ? Cast : Literal;

  bool InMacro = Expression->getBeginLoc().isMacroID();

  if (InMacro && IgnoreMacros)
    return;

  auto Diag =
      diag(Expression->getExprLoc(),
           "converting integer literal to bool, use bool literal instead");

  if (!InMacro)
    Diag << FixItHint::CreateReplacement(
        Expression->getSourceRange(), LiteralBooleanValue ? "true" : "false");
}

} // namespace clang::tidy::modernize
