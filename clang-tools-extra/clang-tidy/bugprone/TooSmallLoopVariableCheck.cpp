//===--- TooSmallLoopVariableCheck.cpp - clang-tidy -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TooSmallLoopVariableCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace bugprone {

static constexpr llvm::StringLiteral LoopName =
    llvm::StringLiteral("forLoopName");
static constexpr llvm::StringLiteral LoopVarName =
    llvm::StringLiteral("loopVar");
static constexpr llvm::StringLiteral LoopVarCastName =
    llvm::StringLiteral("loopVarCast");
static constexpr llvm::StringLiteral LoopUpperBoundName =
    llvm::StringLiteral("loopUpperBound");
static constexpr llvm::StringLiteral LoopIncrementName =
    llvm::StringLiteral("loopIncrement");

/// \brief The matcher for loops with suspicious integer loop variable.
///
/// In this general example, assuming 'j' and 'k' are of integral type:
/// \code
///   for (...; j < 3 + 2; ++k) { ... }
/// \endcode
/// The following string identifiers are bound to these parts of the AST:
///   LoopVarName: 'j' (as a VarDecl)
///   LoopVarCastName: 'j' (after implicit conversion)
///   LoopUpperBoundName: '3 + 2' (as an Expr)
///   LoopIncrementName: 'k' (as an Expr)
///   LoopName: The entire for loop (as a ForStmt)
///
void TooSmallLoopVariableCheck::registerMatchers(MatchFinder *Finder) {
  StatementMatcher LoopVarMatcher =
      expr(
          ignoringParenImpCasts(declRefExpr(to(varDecl(hasType(isInteger()))))))
          .bind(LoopVarName);

  // We need to catch only those comparisons which contain any integer cast.
  StatementMatcher LoopVarConversionMatcher =
      implicitCastExpr(hasImplicitDestinationType(isInteger()),
                       has(ignoringParenImpCasts(LoopVarMatcher)))
          .bind(LoopVarCastName);

  // We are interested in only those cases when the loop bound is a variable
  // value (not const, enum, etc.).
  StatementMatcher LoopBoundMatcher =
      expr(ignoringParenImpCasts(allOf(hasType(isInteger()),
                                       unless(integerLiteral()),
                                       unless(hasType(isConstQualified())),
                                       unless(hasType(enumType())))))
          .bind(LoopUpperBoundName);

  // We use the loop increment expression only to make sure we found the right
  // loop variable.
  StatementMatcher IncrementMatcher =
      expr(ignoringParenImpCasts(hasType(isInteger()))).bind(LoopIncrementName);

  Finder->addMatcher(
      forStmt(
          hasCondition(anyOf(
              binaryOperator(hasOperatorName("<"),
                             hasLHS(LoopVarConversionMatcher),
                             hasRHS(LoopBoundMatcher)),
              binaryOperator(hasOperatorName("<="),
                             hasLHS(LoopVarConversionMatcher),
                             hasRHS(LoopBoundMatcher)),
              binaryOperator(hasOperatorName(">"), hasLHS(LoopBoundMatcher),
                             hasRHS(LoopVarConversionMatcher)),
              binaryOperator(hasOperatorName(">="), hasLHS(LoopBoundMatcher),
                             hasRHS(LoopVarConversionMatcher)))),
          hasIncrement(IncrementMatcher))
          .bind(LoopName),
      this);
}

/// Returns the positive part of the integer width for an integer type.
static unsigned calcPositiveBits(const ASTContext &Context,
                                 const QualType &IntExprType) {
  assert(IntExprType->isIntegerType());

  return IntExprType->isUnsignedIntegerType()
             ? Context.getIntWidth(IntExprType)
             : Context.getIntWidth(IntExprType) - 1;
}

/// \brief Calculate the upper bound expression's positive bits, but ignore
/// constant like values to reduce false positives.
static unsigned calcUpperBoundPositiveBits(const ASTContext &Context,
                                           const Expr *UpperBound,
                                           const QualType &UpperBoundType) {
  // Ignore casting caused by constant values inside a binary operator.
  // We are interested in variable values' positive bits.
  if (const auto *BinOperator = dyn_cast<BinaryOperator>(UpperBound)) {
    const Expr *RHSE = BinOperator->getRHS()->IgnoreParenImpCasts();
    const Expr *LHSE = BinOperator->getLHS()->IgnoreParenImpCasts();

    QualType RHSEType = RHSE->getType();
    QualType LHSEType = LHSE->getType();

    if (!RHSEType->isIntegerType() || !LHSEType->isIntegerType())
      return 0;

    bool RHSEIsConstantValue = RHSEType->isEnumeralType() ||
                               RHSEType.isConstQualified() ||
                               isa<IntegerLiteral>(RHSE);
    bool LHSEIsConstantValue = LHSEType->isEnumeralType() ||
                               LHSEType.isConstQualified() ||
                               isa<IntegerLiteral>(LHSE);

    // Avoid false positives produced by two constant values.
    if (RHSEIsConstantValue && LHSEIsConstantValue)
      return 0;
    if (RHSEIsConstantValue)
      return calcPositiveBits(Context, LHSEType);
    if (LHSEIsConstantValue)
      return calcPositiveBits(Context, RHSEType);

    return std::max(calcPositiveBits(Context, LHSEType),
                    calcPositiveBits(Context, RHSEType));
  }

  return calcPositiveBits(Context, UpperBoundType);
}

void TooSmallLoopVariableCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *LoopVar = Result.Nodes.getNodeAs<Expr>(LoopVarName);
  const auto *UpperBound =
      Result.Nodes.getNodeAs<Expr>(LoopUpperBoundName)->IgnoreParenImpCasts();
  const auto *LoopIncrement =
      Result.Nodes.getNodeAs<Expr>(LoopIncrementName)->IgnoreParenImpCasts();

  // We matched the loop variable incorrectly.
  if (LoopVar->getType() != LoopIncrement->getType())
    return;

  QualType LoopVarType = LoopVar->getType();
  QualType UpperBoundType = UpperBound->getType();

  ASTContext &Context = *Result.Context;

  unsigned LoopVarPosBits = calcPositiveBits(Context, LoopVarType);
  unsigned UpperBoundPosBits =
      calcUpperBoundPositiveBits(Context, UpperBound, UpperBoundType);

  if (UpperBoundPosBits == 0)
    return;

  if (LoopVarPosBits < UpperBoundPosBits)
    diag(LoopVar->getBeginLoc(), "loop variable has narrower type %0 than "
                                 "iteration's upper bound %1")
        << LoopVarType << UpperBoundType;
}

} // namespace bugprone
} // namespace tidy
} // namespace clang
