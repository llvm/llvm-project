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

  const auto RedundantTemporaryString =
      expr(anyOf(RedundantStringConstruction, RedundantFunctionalCast));

  // Matches std::string(...).[c_str()|.data()]
  const auto RedundantStringWithCStr =
      cxxMemberCallExpr(callee(cxxMethodDecl(hasAnyName("c_str", "data"))),
                        on(ignoringParenImpCasts(RedundantTemporaryString)));

  // Main matcher: finds cases where an expression convertible to
  // std::string_view is first converted to std::string unnecessarily.
  Finder->addMatcher(
      cxxMemberCallExpr(
          callee(memberExpr(member(cxxConversionDecl(returns(IsStdStringView))),
                            has(ignoringImplicit(RedundantTemporaryString.bind(
                                "redundantExpr"))))))
          .bind("stringView"),
      this);

  Finder->addMatcher(
      cxxConstructExpr(
          argumentCountIs(1),
          hasArgument(0, RedundantStringWithCStr.bind("redundantExpr")))
          .bind("stringView"),
      this);
}

void StringViewConversionsCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *StringView = Result.Nodes.getNodeAs<Expr>("stringView");
  const auto *RedundantExpr = Result.Nodes.getNodeAs<Expr>("redundantExpr");
  const auto *OriginalExpr = Result.Nodes.getNodeAs<Expr>("originalStringView");
  assert(StringView && RedundantExpr && OriginalExpr);

  bool IsCStrPattern = false;
  StringRef MethodName;
  const auto *CStrCall = dyn_cast<CXXMemberCallExpr>(RedundantExpr);
  if (CStrCall && CStrCall->getMethodDecl()) {
    MethodName = CStrCall->getMethodDecl()->getName();
    if (MethodName == "c_str" || MethodName == "data")
      IsCStrPattern = true;
  }

  const StringRef OriginalText = Lexer::getSourceText(
      CharSourceRange::getTokenRange(OriginalExpr->getSourceRange()),
      *Result.SourceManager, getLangOpts());

  if (OriginalText.empty())
    return;

  const FixItHint FixRedundantConversion = FixItHint::CreateReplacement(
      RedundantExpr->getSourceRange(), OriginalText);
  if (IsCStrPattern && CStrCall) {
    // Handle std::string(sv).c_str() or std::string(sv).data() pattern
    diag(RedundantExpr->getBeginLoc(),
         "redundant conversion to %0 and calling .%1() and then back to %2")
        << CStrCall->getImplicitObjectArgument()->getType() << MethodName
        << StringView->getType() << FixRedundantConversion;
  } else {
    // Handle direct std::string(sv) pattern
    diag(RedundantExpr->getBeginLoc(),
         "redundant conversion to %0 and then back to %1")
        << RedundantExpr->getType() << StringView->getType()
        << FixRedundantConversion;
  }
}

} // namespace clang::tidy::performance
