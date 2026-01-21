//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "StringViewConversionsCheck.h"
#include "clang/AST/Expr.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::performance {

static auto getStringTypeMatcher(StringRef CharType) {
  return hasCanonicalType(hasDeclaration(cxxRecordDecl(hasName(CharType))));
}

void StringViewConversionsCheck::registerMatchers(MatchFinder *Finder) {
  // Matchers for std::basic_[w|u8|u16|u32]string[_view] families.
  const auto IsStdString = getStringTypeMatcher("::std::basic_string");
  const auto IsStdStringView = getStringTypeMatcher("::std::basic_string_view");

  // Matches pointer to any character type (char*, etc.) or array of any
  // character type (char[], etc.).
  const auto IsCharPointerOrArray =
      anyOf(hasType(pointerType(pointee(isAnyCharacter()))),
            hasType(arrayType(hasElementType(isAnyCharacter()))));

  const auto ImplicitlyConvertibleToStringView =
      expr(anyOf(hasType(IsStdStringView), IsCharPointerOrArray,
                 hasType(IsStdString)))
          .bind("originalStringView");

  // Matches std::string construction from a string_view-convertible expression:
  //   - Direct construction: std::string{sv}, std::string{s}
  //   - Copy from existing string: std::string(s) where s is std::string
  const auto RedundantStringConstruction = cxxConstructExpr(
      hasType(IsStdString),
      hasArgument(0, ignoringImplicit(ImplicitlyConvertibleToStringView)),
      unless(hasArgument(1, unless(cxxDefaultArgExpr()))));

  // Matches functional cast syntax: std::string(expr):
  // std::string(sv), std::string("literal")
  const auto RedundantFunctionalCast = cxxFunctionalCastExpr(
      hasType(IsStdString), hasDescendant(RedundantStringConstruction));

  // Match method calls on std::string that modify or use the string,
  // such as operator+, append(), substr(), c_str(), etc.
  const auto HasStringOperatorCall = hasDescendant(cxxOperatorCallExpr(
      hasOverloadedOperatorName("+"), hasType(IsStdString)));
  const auto HasStringMethodCall =
      hasDescendant(cxxMemberCallExpr(on(hasType(IsStdString))));

  const auto IsCallReturningString = callExpr(hasType(IsStdString));
  const auto IsImplicitStringViewFromCall =
      cxxConstructExpr(hasType(IsStdStringView),
                       hasArgument(0, ignoringImplicit(IsCallReturningString)));

  // Main matcher: finds function calls where:
  // 1. A parameter has type string_view
  // 2. The corresponding argument contains a redundant std::string construction
  //    (either functional cast syntax or direct construction/brace init)
  // 3. The argument does NOT involve:
  //    - String concatenation with operator+ (string_view doesn't support it)
  //    - Method calls on the std::string (like append(), substr(), etc.)
  Finder->addMatcher(
      callExpr(forEachArgumentWithParam(
          expr(hasType(IsStdStringView),
               // Ignore cases where the argument is a function call
               unless(ignoringParenImpCasts(IsImplicitStringViewFromCall)),
               // Match either syntax for std::string construction
               hasDescendant(expr(anyOf(RedundantFunctionalCast,
                                        RedundantStringConstruction))
                                 .bind("redundantExpr")),
               // Exclude cases of std::string methods or operator+ calls
               unless(anyOf(HasStringOperatorCall, HasStringMethodCall)))
              .bind("expr"),
          parmVarDecl(hasType(IsStdStringView)))),
      this);
}

void StringViewConversionsCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *ParamExpr = Result.Nodes.getNodeAs<Expr>("expr");
  const auto *RedundantExpr = Result.Nodes.getNodeAs<Expr>("redundantExpr");
  const auto *OriginalExpr = Result.Nodes.getNodeAs<Expr>("originalStringView");
  assert(RedundantExpr && ParamExpr && OriginalExpr);

  // Sanity check. Verify that the redundant expression is the direct source of
  // the argument, not part of a larger expression (e.g., std::string(sv) +
  // "bar").
  assert(ParamExpr->getSourceRange() == RedundantExpr->getSourceRange());

  const StringRef OriginalText = Lexer::getSourceText(
      CharSourceRange::getTokenRange(OriginalExpr->getSourceRange()),
      *Result.SourceManager, getLangOpts());

  if (OriginalText.empty())
    return;

  diag(RedundantExpr->getBeginLoc(),
       "redundant conversion to %0 and then back to %1")
      << RedundantExpr->getType() << ParamExpr->getType()
      << FixItHint::CreateReplacement(RedundantExpr->getSourceRange(),
                                      OriginalText);
}

} // namespace clang::tidy::performance
