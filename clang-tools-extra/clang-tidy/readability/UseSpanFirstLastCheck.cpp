//===----------------------------------------------------------------------===//
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
  // Type matcher for concrete std::span types.
  const auto HasSpanType =
      hasType(hasUnqualifiedDesugaredType(recordType(hasDeclaration(
          classTemplateSpecializationDecl(hasName("::std::span"))))));

  // Type matcher for dependent std::span types (in templates).
  const auto HasDependentSpanType = hasType(
      hasCanonicalType(hasDeclaration(namedDecl(hasName("::std::span")))));

  // --- Non-dependent matchers (concrete types) ---

  // Match span.subspan(0, n) -> first(n)
  Finder->addMatcher(
      cxxMemberCallExpr(
          argumentCountIs(2),
          callee(memberExpr(hasDeclaration(cxxMethodDecl(hasName("subspan"))))),
          on(expr(HasSpanType).bind("span_object")),
          hasArgument(0, integerLiteral(equals(0))),
          hasArgument(1, expr().bind("count")))
          .bind("subspan_call"),
      this);

  // Match span.subspan(span.size() - n) -> last(n)
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
          .bind("subspan_call"),
      this);

  // --- Dependent matchers (template definitions) ---

  const auto DependentSubspanCallee = callee(cxxDependentScopeMemberExpr(
      hasMemberName("subspan"),
      hasObjectExpression(
          expr(anyOf(HasDependentSpanType, HasSpanType)).bind("span_object"))));

  // Match span.subspan(0, n) -> first(n) in dependent context
  Finder->addMatcher(callExpr(argumentCountIs(2), DependentSubspanCallee,
                              hasArgument(0, integerLiteral(equals(0))),
                              hasArgument(1, expr().bind("count")))
                         .bind("subspan_call"),
                     this);

  // Match span.subspan(span.size() - n) -> last(n) in dependent context
  const auto DependentSizeCall = callExpr(callee(cxxDependentScopeMemberExpr(
      hasMemberName("size"),
      hasObjectExpression(
          expr(isStatementIdenticalToBoundNode("span_object"))))));

  Finder->addMatcher(
      callExpr(argumentCountIs(1), DependentSubspanCallee,
               hasArgument(0, binaryOperator(hasOperatorName("-"),
                                             hasLHS(DependentSizeCall),
                                             hasRHS(expr().bind("count")))))
          .bind("subspan_call"),
      this);
}

void UseSpanFirstLastCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *SpanObj = Result.Nodes.getNodeAs<Expr>("span_object");
  if (!SpanObj)
    return;

  const auto *SubSpan = Result.Nodes.getNodeAs<CallExpr>("subspan_call");
  if (!SubSpan)
    return;

  const bool IsFirst = SubSpan->getNumArgs() == 2;

  const auto *Count = Result.Nodes.getNodeAs<Expr>("count");
  assert(Count && "Count expression must exist due to AST matcher");

  const StringRef CountText = Lexer::getSourceText(
      CharSourceRange::getTokenRange(Count->getSourceRange()),
      *Result.SourceManager, Result.Context->getLangOpts());
  const StringRef SpanText = Lexer::getSourceText(
      CharSourceRange::getTokenRange(SpanObj->getSourceRange()),
      *Result.SourceManager, Result.Context->getLangOpts());
  const StringRef FirstOrLast = IsFirst ? "first" : "last";
  const std::string Replacement =
      (Twine(SpanText) + "." + FirstOrLast + "(" + CountText + ")").str();

  diag(SubSpan->getBeginLoc(), "prefer 'span::%0()' over 'subspan()'")
      << FirstOrLast
      << FixItHint::CreateReplacement(SubSpan->getSourceRange(), Replacement);
}
} // namespace clang::tidy::readability
