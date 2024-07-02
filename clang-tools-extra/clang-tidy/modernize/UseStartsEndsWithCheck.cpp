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
          hasCanonicalType(hasDeclaration(ClassWithStartsWithFunction)))),
      // Bind search expression.
      hasArgument(0, expr().bind("search_expr")));

  const auto RFindExpr = cxxMemberCallExpr(
      // A method call with a second argument of zero...
      hasArgument(1, ZeroLiteral),
      // ... named rfind...
      callee(cxxMethodDecl(hasName("rfind")).bind("find_fun")),
      // ... on a class with a starts_with function.
      on(hasType(
          hasCanonicalType(hasDeclaration(ClassWithStartsWithFunction)))),
      // Bind search expression.
      hasArgument(0, expr().bind("search_expr")));

  // Match a string literal and an integer or strlen() call matching the length.
  const auto HasStringLiteralAndLengthArgs = [](const auto StringArgIndex,
                                                const auto LengthArgIndex) {
    return allOf(
        hasArgument(StringArgIndex, stringLiteral().bind("string_literal_arg")),
        hasArgument(LengthArgIndex,
                    anyOf(integerLiteral().bind("integer_literal_size_arg"),
                          callExpr(callee(functionDecl(parameterCountIs(1),
                                                       hasName("strlen"))),
                                   hasArgument(0, stringLiteral().bind(
                                                      "strlen_arg"))))));
  };

  // Match a string variable and a call to length() or size().
  const auto HasStringVariableAndSizeCallArgs = [](const auto StringArgIndex,
                                                   const auto LengthArgIndex) {
    return allOf(
        hasArgument(StringArgIndex, declRefExpr(hasDeclaration(
                                        decl().bind("string_var_decl")))),
        hasArgument(LengthArgIndex,
                    cxxMemberCallExpr(
                        callee(cxxMethodDecl(isConst(), parameterCountIs(0),
                                             hasAnyName("size", "length"))),
                        on(declRefExpr(
                            to(decl(equalsBoundNode("string_var_decl"))))))));
  };

  // Match either one of the two cases above.
  const auto HasStringAndLengthArgs =
      [HasStringLiteralAndLengthArgs, HasStringVariableAndSizeCallArgs](
          const auto StringArgIndex, const auto LengthArgIndex) {
        return anyOf(
            HasStringLiteralAndLengthArgs(StringArgIndex, LengthArgIndex),
            HasStringVariableAndSizeCallArgs(StringArgIndex, LengthArgIndex));
      };

  const auto CompareExpr = cxxMemberCallExpr(
      // A method call with three arguments...
      argumentCountIs(3),
      // ... where the first argument is zero...
      hasArgument(0, ZeroLiteral),
      // ... named compare...
      callee(cxxMethodDecl(hasName("compare")).bind("find_fun")),
      // ... on a class with a starts_with function...
      on(hasType(
          hasCanonicalType(hasDeclaration(ClassWithStartsWithFunction)))),
      // ... where the third argument is some string and the second a length.
      HasStringAndLengthArgs(2, 1),
      // Bind search expression.
      hasArgument(2, expr().bind("search_expr")));

  Finder->addMatcher(
      // Match [=!]= with a zero on one side and (r?)find|compare on the other.
      binaryOperator(
          hasAnyOperatorName("==", "!="),
          hasOperands(cxxMemberCallExpr(anyOf(FindExpr, RFindExpr, CompareExpr))
                          .bind("find_expr"),
                      ZeroLiteral))
          .bind("expr"),
      this);
}

void UseStartsEndsWithCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *ComparisonExpr = Result.Nodes.getNodeAs<BinaryOperator>("expr");
  const auto *FindExpr = Result.Nodes.getNodeAs<CXXMemberCallExpr>("find_expr");
  const auto *FindFun = Result.Nodes.getNodeAs<CXXMethodDecl>("find_fun");
  const auto *SearchExpr = Result.Nodes.getNodeAs<Expr>("search_expr");
  const auto *StartsWithFunction =
      Result.Nodes.getNodeAs<CXXMethodDecl>("starts_with_fun");

  const auto *StringLiteralArg =
      Result.Nodes.getNodeAs<StringLiteral>("string_literal_arg");
  const auto *IntegerLiteralSizeArg =
      Result.Nodes.getNodeAs<IntegerLiteral>("integer_literal_size_arg");
  const auto *StrlenArg = Result.Nodes.getNodeAs<StringLiteral>("strlen_arg");

  // Filter out compare cases where the length does not match string literal.
  if (StringLiteralArg && IntegerLiteralSizeArg &&
      StringLiteralArg->getLength() !=
          IntegerLiteralSizeArg->getValue().getZExtValue()) {
    return;
  }

  if (StringLiteralArg && StrlenArg &&
      StringLiteralArg->getLength() != StrlenArg->getLength()) {
    return;
  }

  if (ComparisonExpr->getBeginLoc().isMacroID()) {
    return;
  }

  const bool Neg = ComparisonExpr->getOpcode() == BO_NE;

  auto Diagnostic =
      diag(FindExpr->getExprLoc(), "use %0 instead of %1() %select{==|!=}2 0")
      << StartsWithFunction->getName() << FindFun->getName() << Neg;

  // Remove possible arguments after search expression and ' [!=]= 0' suffix.
  Diagnostic << FixItHint::CreateReplacement(
      CharSourceRange::getTokenRange(
          Lexer::getLocForEndOfToken(SearchExpr->getEndLoc(), 0,
                                     *Result.SourceManager, getLangOpts()),
          ComparisonExpr->getEndLoc()),
      ")");

  // Remove possible '0 [!=]= ' prefix.
  Diagnostic << FixItHint::CreateRemoval(CharSourceRange::getCharRange(
      ComparisonExpr->getBeginLoc(), FindExpr->getBeginLoc()));

  // Replace method name by 'starts_with'.
  // Remove possible arguments before search expression.
  Diagnostic << FixItHint::CreateReplacement(
      CharSourceRange::getCharRange(FindExpr->getExprLoc(),
                                    SearchExpr->getBeginLoc()),
      (StartsWithFunction->getName() + "(").str());

  // Add possible negation '!'.
  if (Neg) {
    Diagnostic << FixItHint::CreateInsertion(FindExpr->getBeginLoc(), "!");
  }
}

} // namespace clang::tidy::modernize
