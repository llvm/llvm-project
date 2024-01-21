//===--- UseStartsEndsWithCheck.cpp - clang-tidy --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseStartsEndsWithCheck.h"

#include "../utils/OptionsUtils.h"
#include "clang/Lex/Lexer.h"

#include <string>

using namespace clang::ast_matchers;

namespace clang::tidy::modernize {

UseStartsEndsWithCheck::UseStartsEndsWithCheck(StringRef Name,
                                               ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context) {}

void UseStartsEndsWithCheck::registerMatchers(MatchFinder *Finder) {
  const auto ZeroLiteral = integerLiteral(equals(0));
  const auto HasStartsWithMethodWithName = [](const std::string &Name) {
    return hasMethod(
        cxxMethodDecl(hasName(Name), isConst(), parameterCountIs(1))
            .bind("starts_with_fun"));
  };
  const auto HasStartsWithMethod =
      anyOf(HasStartsWithMethodWithName("starts_with"),
            HasStartsWithMethodWithName("startsWith"),
            HasStartsWithMethodWithName("startswith"));
  const auto ClassWithStartsWithFunction = cxxRecordDecl(anyOf(
      HasStartsWithMethod, hasAnyBase(hasType(hasCanonicalType(hasDeclaration(
                               cxxRecordDecl(HasStartsWithMethod)))))));

  const auto FindExpr = cxxMemberCallExpr(
      // A method call with no second argument or the second argument is zero...
      anyOf(argumentCountIs(1), hasArgument(1, ZeroLiteral)),
      // ... named find...
      callee(cxxMethodDecl(hasName("find")).bind("find_fun")),
      // ... on a class with a starts_with function.
      on(hasType(
          hasCanonicalType(hasDeclaration(ClassWithStartsWithFunction)))));

  const auto RFindExpr = cxxMemberCallExpr(
      // A method call with a second argument of zero...
      hasArgument(1, ZeroLiteral),
      // ... named rfind...
      callee(cxxMethodDecl(hasName("rfind")).bind("find_fun")),
      // ... on a class with a starts_with function.
      on(hasType(
          hasCanonicalType(hasDeclaration(ClassWithStartsWithFunction)))));

  const auto FindOrRFindExpr =
      cxxMemberCallExpr(anyOf(FindExpr, RFindExpr)).bind("find_expr");

  Finder->addMatcher(
      // Match [=!]= with a zero on one side and a string.(r?)find on the other.
      binaryOperator(hasAnyOperatorName("==", "!="),
                     hasOperands(FindOrRFindExpr, ZeroLiteral))
          .bind("expr"),
      this);
}

void UseStartsEndsWithCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *ComparisonExpr = Result.Nodes.getNodeAs<BinaryOperator>("expr");
  const auto *FindExpr = Result.Nodes.getNodeAs<CXXMemberCallExpr>("find_expr");
  const auto *FindFun = Result.Nodes.getNodeAs<CXXMethodDecl>("find_fun");
  const auto *StartsWithFunction =
      Result.Nodes.getNodeAs<CXXMethodDecl>("starts_with_fun");

  if (ComparisonExpr->getBeginLoc().isMacroID()) {
    return;
  }

  const bool Neg = ComparisonExpr->getOpcode() == BO_NE;

  auto Diagnostic =
      diag(FindExpr->getBeginLoc(), "use %0 instead of %1() %select{==|!=}2 0")
      << StartsWithFunction->getName() << FindFun->getName() << Neg;

  // Remove possible zero second argument and ' [!=]= 0' suffix.
  Diagnostic << FixItHint::CreateReplacement(
      CharSourceRange::getTokenRange(
          Lexer::getLocForEndOfToken(FindExpr->getArg(0)->getEndLoc(), 0,
                                     *Result.SourceManager, getLangOpts()),
          ComparisonExpr->getEndLoc()),
      ")");

  // Remove possible '0 [!=]= ' prefix.
  Diagnostic << FixItHint::CreateRemoval(CharSourceRange::getCharRange(
      ComparisonExpr->getBeginLoc(), FindExpr->getBeginLoc()));

  // Replace '(r?)find' with 'starts_with'.
  Diagnostic << FixItHint::CreateReplacement(
      CharSourceRange::getTokenRange(FindExpr->getExprLoc(),
                                     FindExpr->getExprLoc()),
      StartsWithFunction->getName());

  // Add possible negation '!'.
  if (Neg) {
    Diagnostic << FixItHint::CreateInsertion(FindExpr->getBeginLoc(), "!");
  }
}

} // namespace clang::tidy::modernize
