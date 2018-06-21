//===--- NarrowingConversionsCheck.cpp - clang-tidy------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "NarrowingConversionsCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace cppcoreguidelines {

// FIXME: Check double -> float truncation. Pay attention to casts:
void NarrowingConversionsCheck::registerMatchers(MatchFinder *Finder) {
  // ceil() and floor() are guaranteed to return integers, even though the type
  // is not integral.
  const auto IsCeilFloorCall = callExpr(callee(functionDecl(
      hasAnyName("::ceil", "::std::ceil", "::floor", "::std::floor"))));

  const auto IsFloatExpr =
      expr(hasType(realFloatingPointType()), unless(IsCeilFloorCall));

  // casts:
  //   i = 0.5;
  //   void f(int); f(0.5);
  Finder->addMatcher(implicitCastExpr(hasImplicitDestinationType(isInteger()),
                                      hasSourceExpression(IsFloatExpr),
                                      unless(hasParent(castExpr())),
                                      unless(isInTemplateInstantiation()))
                         .bind("cast"),
                     this);

  // Binary operators:
  //   i += 0.5;
  Finder->addMatcher(
      binaryOperator(isAssignmentOperator(),
                     // The `=` case generates an implicit cast which is covered
                     // by the previous matcher.
                     unless(hasOperatorName("=")),
                     hasLHS(hasType(isInteger())), hasRHS(IsFloatExpr),
                     unless(isInTemplateInstantiation()))
          .bind("op"),
      this);
}

void NarrowingConversionsCheck::check(const MatchFinder::MatchResult &Result) {
  if (const auto *Op = Result.Nodes.getNodeAs<BinaryOperator>("op")) {
    if (Op->getLocStart().isMacroID())
      return;
    diag(Op->getOperatorLoc(), "narrowing conversion from %0 to %1")
        << Op->getRHS()->getType() << Op->getLHS()->getType();
    return;
  }
  const auto *Cast = Result.Nodes.getNodeAs<ImplicitCastExpr>("cast");
  if (Cast->getLocStart().isMacroID())
    return;
  diag(Cast->getExprLoc(), "narrowing conversion from %0 to %1")
      << Cast->getSubExpr()->getType() << Cast->getType();
}

} // namespace cppcoreguidelines
} // namespace tidy
} // namespace clang
