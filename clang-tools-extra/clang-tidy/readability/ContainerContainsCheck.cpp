//===--- ContainerContainsCheck.cpp - clang-tidy --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ContainerContainsCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::readability {

void ContainerContainsCheck::registerMatchers(MatchFinder *Finder) {
  const auto SupportedContainers = hasType(
      hasUnqualifiedDesugaredType(recordType(hasDeclaration(cxxRecordDecl(
          hasAnyName("::std::set", "::std::unordered_set", "::std::map",
                     "::std::unordered_map", "::std::multiset",
                     "::std::unordered_multiset", "::std::multimap",
                     "::std::unordered_multimap"))))));

  const auto CountCall =
      cxxMemberCallExpr(on(SupportedContainers),
                        callee(cxxMethodDecl(hasName("count"))),
                        argumentCountIs(1))
          .bind("call");

  const auto FindCall =
      cxxMemberCallExpr(on(SupportedContainers),
                        callee(cxxMethodDecl(hasName("find"))),
                        argumentCountIs(1))
          .bind("call");

  const auto EndCall = cxxMemberCallExpr(on(SupportedContainers),
                                         callee(cxxMethodDecl(hasName("end"))),
                                         argumentCountIs(0));

  const auto Literal0 = integerLiteral(equals(0));
  const auto Literal1 = integerLiteral(equals(1));

  auto AddSimpleMatcher = [&](auto Matcher) {
    Finder->addMatcher(
        traverse(TK_IgnoreUnlessSpelledInSource, std::move(Matcher)), this);
  };

  // Find membership tests which use `count()`.
  Finder->addMatcher(implicitCastExpr(hasImplicitDestinationType(booleanType()),
                                      hasSourceExpression(CountCall))
                         .bind("positiveComparison"),
                     this);
  AddSimpleMatcher(
      binaryOperator(hasLHS(CountCall), hasOperatorName("!="), hasRHS(Literal0))
          .bind("positiveComparison"));
  AddSimpleMatcher(
      binaryOperator(hasLHS(Literal0), hasOperatorName("!="), hasRHS(CountCall))
          .bind("positiveComparison"));
  AddSimpleMatcher(
      binaryOperator(hasLHS(CountCall), hasOperatorName(">"), hasRHS(Literal0))
          .bind("positiveComparison"));
  AddSimpleMatcher(
      binaryOperator(hasLHS(Literal0), hasOperatorName("<"), hasRHS(CountCall))
          .bind("positiveComparison"));
  AddSimpleMatcher(
      binaryOperator(hasLHS(CountCall), hasOperatorName(">="), hasRHS(Literal1))
          .bind("positiveComparison"));
  AddSimpleMatcher(
      binaryOperator(hasLHS(Literal1), hasOperatorName("<="), hasRHS(CountCall))
          .bind("positiveComparison"));

  // Find inverted membership tests which use `count()`.
  AddSimpleMatcher(
      binaryOperator(hasLHS(CountCall), hasOperatorName("=="), hasRHS(Literal0))
          .bind("negativeComparison"));
  AddSimpleMatcher(
      binaryOperator(hasLHS(Literal0), hasOperatorName("=="), hasRHS(CountCall))
          .bind("negativeComparison"));
  AddSimpleMatcher(
      binaryOperator(hasLHS(CountCall), hasOperatorName("<="), hasRHS(Literal0))
          .bind("negativeComparison"));
  AddSimpleMatcher(
      binaryOperator(hasLHS(Literal0), hasOperatorName(">="), hasRHS(CountCall))
          .bind("negativeComparison"));
  AddSimpleMatcher(
      binaryOperator(hasLHS(CountCall), hasOperatorName("<"), hasRHS(Literal1))
          .bind("negativeComparison"));
  AddSimpleMatcher(
      binaryOperator(hasLHS(Literal1), hasOperatorName(">"), hasRHS(CountCall))
          .bind("negativeComparison"));

  // Find membership tests based on `find() == end()`.
  AddSimpleMatcher(
      binaryOperator(hasLHS(FindCall), hasOperatorName("!="), hasRHS(EndCall))
          .bind("positiveComparison"));
  AddSimpleMatcher(
      binaryOperator(hasLHS(FindCall), hasOperatorName("=="), hasRHS(EndCall))
          .bind("negativeComparison"));
}

void ContainerContainsCheck::check(const MatchFinder::MatchResult &Result) {
  // Extract the information about the match
  const auto *Call = Result.Nodes.getNodeAs<CXXMemberCallExpr>("call");
  const auto *PositiveComparison =
      Result.Nodes.getNodeAs<Expr>("positiveComparison");
  const auto *NegativeComparison =
      Result.Nodes.getNodeAs<Expr>("negativeComparison");
  assert((!PositiveComparison || !NegativeComparison) &&
         "only one of PositiveComparison or NegativeComparison should be set");
  bool Negated = NegativeComparison != nullptr;
  const auto *Comparison = Negated ? NegativeComparison : PositiveComparison;

  // Diagnose the issue.
  auto Diag =
      diag(Call->getExprLoc(), "use 'contains' to check for membership");

  // Don't fix it if it's in a macro invocation. Leave fixing it to the user.
  SourceLocation FuncCallLoc = Comparison->getEndLoc();
  if (!FuncCallLoc.isValid() || FuncCallLoc.isMacroID())
    return;

  // Create the fix it.
  const auto *Member = cast<MemberExpr>(Call->getCallee());
  Diag << FixItHint::CreateReplacement(
      Member->getMemberNameInfo().getSourceRange(), "contains");
  SourceLocation ComparisonBegin = Comparison->getSourceRange().getBegin();
  SourceLocation ComparisonEnd = Comparison->getSourceRange().getEnd();
  SourceLocation CallBegin = Call->getSourceRange().getBegin();
  SourceLocation CallEnd = Call->getSourceRange().getEnd();
  Diag << FixItHint::CreateReplacement(
      CharSourceRange::getCharRange(ComparisonBegin, CallBegin),
      Negated ? "!" : "");
  Diag << FixItHint::CreateRemoval(CharSourceRange::getTokenRange(
      CallEnd.getLocWithOffset(1), ComparisonEnd));
}

} // namespace clang::tidy::readability
