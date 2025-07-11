//===--- UseSpanFirstLastCheck.cpp - clang-tidy -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseSpanFirstLastCheck.h"
#include "../utils/Matchers.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;
using namespace clang::tidy::matchers;

namespace clang::tidy::readability {

void UseSpanFirstLastCheck::registerMatchers(MatchFinder *Finder) {
  const auto HasSpanType =
      hasType(hasUnqualifiedDesugaredType(recordType(hasDeclaration(
          classTemplateSpecializationDecl(hasName("::std::span"))))));

  const auto SubspanDecl = cxxMethodDecl(
      hasName("subspan"),
      ofClass(classTemplateSpecializationDecl(hasName("::std::span"))));

  // Match span.subspan(0, n) -> first(n)
  Finder->addMatcher(
      cxxMemberCallExpr(
          argumentCountIs(2),
          callee(memberExpr(hasDeclaration(cxxMethodDecl(hasName("subspan"))))),
          on(expr(HasSpanType).bind("span_object")),
          hasArgument(0, integerLiteral(equals(0))),
          hasArgument(1, expr().bind("count")))
          .bind("first_subspan"),
      this);

  // Match span.subspan(span.size() - n) or span.subspan(std::ranges::size(span)
  // - n)
  // -> last(n)
  const auto SizeCall = anyOf(
      cxxMemberCallExpr(
          callee(memberExpr(hasDeclaration(cxxMethodDecl(hasName("size"))))),
          on(expr(isStatementIdenticalToBoundNode("span_object")))),
      callExpr(callee(functionDecl(
                   hasAnyName("::std::size", "::std::ranges::size"))),
               hasArgument(
                   0, expr(isStatementIdenticalToBoundNode("span_object")))));

  Finder->addMatcher(
      cxxMemberCallExpr(
          argumentCountIs(1),
          callee(memberExpr(hasDeclaration(cxxMethodDecl(hasName("subspan"))))),
          on(expr(HasSpanType).bind("span_object")),
          hasArgument(0, binaryOperator(hasOperatorName("-"), hasLHS(SizeCall),
                                        hasRHS(expr().bind("count")))))
          .bind("last_subspan"),
      this);
}

void UseSpanFirstLastCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *SpanObj = Result.Nodes.getNodeAs<Expr>("span_object");
  if (!SpanObj)
    return;

  const auto *SubSpan =
      Result.Nodes.getNodeAs<CXXMemberCallExpr>("first_subspan");
  bool IsFirst = true;
  if (!SubSpan) {
    SubSpan = Result.Nodes.getNodeAs<CXXMemberCallExpr>("last_subspan");
    IsFirst = false;
  }

  if (!SubSpan)
    return;

  const auto *Count = Result.Nodes.getNodeAs<Expr>("count");
  assert(Count && "Count expression must exist due to AST matcher");

  StringRef CountText = Lexer::getSourceText(
      CharSourceRange::getTokenRange(Count->getSourceRange()),
      *Result.SourceManager, Result.Context->getLangOpts());
  StringRef SpanText = Lexer::getSourceText(
      CharSourceRange::getTokenRange(SpanObj->getSourceRange()),
      *Result.SourceManager, Result.Context->getLangOpts());
  const StringRef FirstOrLast = IsFirst ? "first" : "last";
  std::string Replacement =
      (Twine(SpanText) + "." + FirstOrLast + "(" + CountText + ")").str();

  diag(SubSpan->getBeginLoc(), "prefer 'span::%0()' over 'subspan()'")
      << FirstOrLast
      << FixItHint::CreateReplacement(SubSpan->getSourceRange(), Replacement);
}
} // namespace clang::tidy::readability
