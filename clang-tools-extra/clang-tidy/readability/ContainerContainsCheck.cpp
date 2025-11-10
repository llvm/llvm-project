//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ContainerContainsCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::readability {
void ContainerContainsCheck::registerMatchers(MatchFinder *Finder) {
  const auto Literal0 = integerLiteral(equals(0));
  const auto Literal1 = integerLiteral(equals(1));

  const auto ClassWithContains = cxxRecordDecl(
      hasMethod(cxxMethodDecl(isConst(), parameterCountIs(1), isPublic(),
                              unless(isDeleted()), returns(booleanType()),
                              hasAnyName("contains", "Contains"))
                    .bind("contains_fun")));

  const auto CountCall =
      cxxMemberCallExpr(argumentCountIs(1),
                        callee(cxxMethodDecl(hasAnyName("count", "Count"),
                                             ofClass(ClassWithContains))))
          .bind("call");

  const auto FindCall =
      // Either one argument, or assume the second argument is the position to
      // start searching from.
      cxxMemberCallExpr(
          anyOf(argumentCountIs(1),
                allOf(argumentCountIs(2), hasArgument(1, Literal0))),
          callee(cxxMethodDecl(hasAnyName("find", "Find"),
                               ofClass(ClassWithContains))))
          .bind("call");

  const auto EndCall = cxxMemberCallExpr(
      argumentCountIs(0), callee(cxxMethodDecl(hasAnyName("end", "End"),
                                               ofClass(ClassWithContains))));

  const auto StringNpos = anyOf(declRefExpr(to(varDecl(hasName("npos")))),
                                memberExpr(member(hasName("npos"))));

  auto AddSimpleMatcher = [&](const auto &Matcher) {
    Finder->addMatcher(traverse(TK_IgnoreUnlessSpelledInSource, Matcher), this);
  };

  // Find membership tests which use `count()`.
  Finder->addMatcher(implicitCastExpr(hasImplicitDestinationType(booleanType()),
                                      hasSourceExpression(CountCall))
                         .bind("positiveComparison"),
                     this);
  AddSimpleMatcher(
      binaryOperation(hasOperatorName("!="), hasOperands(CountCall, Literal0))
          .bind("positiveComparison"));
  AddSimpleMatcher(
      binaryOperation(hasLHS(CountCall), hasOperatorName(">"), hasRHS(Literal0))
          .bind("positiveComparison"));
  AddSimpleMatcher(
      binaryOperation(hasLHS(Literal0), hasOperatorName("<"), hasRHS(CountCall))
          .bind("positiveComparison"));
  AddSimpleMatcher(binaryOperation(hasLHS(CountCall), hasOperatorName(">="),
                                   hasRHS(Literal1))
                       .bind("positiveComparison"));
  AddSimpleMatcher(binaryOperation(hasLHS(Literal1), hasOperatorName("<="),
                                   hasRHS(CountCall))
                       .bind("positiveComparison"));

  // Find inverted membership tests which use `count()`.
  AddSimpleMatcher(
      binaryOperation(hasOperatorName("=="), hasOperands(CountCall, Literal0))
          .bind("negativeComparison"));
  AddSimpleMatcher(binaryOperation(hasLHS(CountCall), hasOperatorName("<="),
                                   hasRHS(Literal0))
                       .bind("negativeComparison"));
  AddSimpleMatcher(binaryOperation(hasLHS(Literal0), hasOperatorName(">="),
                                   hasRHS(CountCall))
                       .bind("negativeComparison"));
  AddSimpleMatcher(
      binaryOperation(hasLHS(CountCall), hasOperatorName("<"), hasRHS(Literal1))
          .bind("negativeComparison"));
  AddSimpleMatcher(
      binaryOperation(hasLHS(Literal1), hasOperatorName(">"), hasRHS(CountCall))
          .bind("negativeComparison"));

  // Find membership tests based on `find() == end()` or `find() == npos`.
  AddSimpleMatcher(
      binaryOperation(hasOperatorName("!="),
                      hasOperands(FindCall, anyOf(EndCall, StringNpos)))
          .bind("positiveComparison"));
  AddSimpleMatcher(
      binaryOperation(hasOperatorName("=="),
                      hasOperands(FindCall, anyOf(EndCall, StringNpos)))
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
  const bool Negated = NegativeComparison != nullptr;
  const auto *Comparison = Negated ? NegativeComparison : PositiveComparison;
  const StringRef ContainsFunName =
      Result.Nodes.getNodeAs<CXXMethodDecl>("contains_fun")->getName();
  const Expr *SearchExpr = Call->getArg(0)->IgnoreParenImpCasts();

  // Diagnose the issue.
  auto Diag = diag(Call->getExprLoc(), "use '%0' to check for membership")
              << ContainsFunName;

  // Don't fix it if it's in a macro invocation. Leave fixing it to the user.
  const SourceLocation FuncCallLoc = Comparison->getEndLoc();
  if (!FuncCallLoc.isValid() || FuncCallLoc.isMacroID())
    return;

  const StringRef SearchExprText = Lexer::getSourceText(
      CharSourceRange::getTokenRange(SearchExpr->getSourceRange()),
      *Result.SourceManager, Result.Context->getLangOpts());

  // Remove everything before the function call.
  Diag << FixItHint::CreateRemoval(CharSourceRange::getCharRange(
      Comparison->getBeginLoc(), Call->getBeginLoc()));

  // Rename the function to `contains`.
  Diag << FixItHint::CreateReplacement(Call->getExprLoc(), ContainsFunName);

  // Replace arguments and everything after the function call.
  Diag << FixItHint::CreateReplacement(
      CharSourceRange::getTokenRange(Call->getArg(0)->getBeginLoc(),
                                     Comparison->getEndLoc()),
      (SearchExprText + ")").str());

  // Add negation if necessary.
  if (Negated)
    Diag << FixItHint::CreateInsertion(Call->getBeginLoc(), "!");
}

} // namespace clang::tidy::readability
