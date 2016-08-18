//=== ConversionChecker.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Check that there is no loss of sign/precision in assignments, comparisons
// and multiplications.
//
// ConversionChecker uses path sensitive analysis to determine possible values
// of expressions. A warning is reported when:
// * a negative value is implicitly converted to an unsigned value in an
//   assignment, comparison or multiplication.
// * assignment / initialization when source value is greater than the max
//   value of target
//
// Many compilers and tools have similar checks that are based on semantic
// analysis. Those checks are sound but have poor precision. ConversionChecker
// is an alternative to those checks.
//
//===----------------------------------------------------------------------===//
#include "ClangSACheckers.h"
#include "clang/AST/ParentMap.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"

using namespace clang;
using namespace ento;

namespace {
class ConversionChecker : public Checker<check::PreStmt<ImplicitCastExpr>> {
public:
  void checkPreStmt(const ImplicitCastExpr *Cast, CheckerContext &C) const;

private:
  mutable std::unique_ptr<BuiltinBug> BT;

  // Is there loss of precision
  bool isLossOfPrecision(const ImplicitCastExpr *Cast, CheckerContext &C) const;

  // Is there loss of sign
  bool isLossOfSign(const ImplicitCastExpr *Cast, CheckerContext &C) const;

  void reportBug(ExplodedNode *N, CheckerContext &C, const char Msg[]) const;
};
}

void ConversionChecker::checkPreStmt(const ImplicitCastExpr *Cast,
                                     CheckerContext &C) const {
  // TODO: For now we only warn about DeclRefExpr, to avoid noise. Warn for
  // calculations also.
  if (!isa<DeclRefExpr>(Cast->IgnoreParenImpCasts()))
    return;

  // Don't warn for loss of sign/precision in macros.
  if (Cast->getExprLoc().isMacroID())
    return;

  // Get Parent.
  const ParentMap &PM = C.getLocationContext()->getParentMap();
  const Stmt *Parent = PM.getParent(Cast);
  if (!Parent)
    return;

  bool LossOfSign = false;
  bool LossOfPrecision = false;

  // Loss of sign/precision in binary operation.
  if (const auto *B = dyn_cast<BinaryOperator>(Parent)) {
    BinaryOperator::Opcode Opc = B->getOpcode();
    if (Opc == BO_Assign || Opc == BO_AddAssign || Opc == BO_SubAssign ||
        Opc == BO_MulAssign) {
      LossOfSign = isLossOfSign(Cast, C);
      LossOfPrecision = isLossOfPrecision(Cast, C);
    } else if (B->isRelationalOp() || B->isMultiplicativeOp()) {
      LossOfSign = isLossOfSign(Cast, C);
    }
  } else if (isa<DeclStmt>(Parent)) {
    LossOfSign = isLossOfSign(Cast, C);
    LossOfPrecision = isLossOfPrecision(Cast, C);
  }

  if (LossOfSign || LossOfPrecision) {
    // Generate an error node.
    ExplodedNode *N = C.generateNonFatalErrorNode(C.getState());
    if (!N)
      return;
    if (LossOfSign)
      reportBug(N, C, "Loss of sign in implicit conversion");
    if (LossOfPrecision)
      reportBug(N, C, "Loss of precision in implicit conversion");
  }
}

void ConversionChecker::reportBug(ExplodedNode *N, CheckerContext &C,
                                  const char Msg[]) const {
  if (!BT)
    BT.reset(
        new BuiltinBug(this, "Conversion", "Possible loss of sign/precision."));

  // Generate a report for this bug.
  auto R = llvm::make_unique<BugReport>(*BT, Msg, N);
  C.emitReport(std::move(R));
}

// Is E value greater or equal than Val?
static bool isGreaterEqual(CheckerContext &C, const Expr *E,
                           unsigned long long Val) {
  ProgramStateRef State = C.getState();
  SVal EVal = C.getSVal(E);
  if (EVal.isUnknownOrUndef() || !EVal.getAs<NonLoc>())
    return false;

  SValBuilder &Bldr = C.getSValBuilder();
  DefinedSVal V = Bldr.makeIntVal(Val, C.getASTContext().LongLongTy);

  // Is DefinedEVal greater or equal with V?
  SVal GE = Bldr.evalBinOp(State, BO_GE, EVal, V, Bldr.getConditionType());
  if (GE.isUnknownOrUndef())
    return false;
  ConstraintManager &CM = C.getConstraintManager();
  ProgramStateRef StGE, StLT;
  std::tie(StGE, StLT) = CM.assumeDual(State, GE.castAs<DefinedSVal>());
  return StGE && !StLT;
}

// Is E value negative?
static bool isNegative(CheckerContext &C, const Expr *E) {
  ProgramStateRef State = C.getState();
  SVal EVal = State->getSVal(E, C.getLocationContext());
  if (EVal.isUnknownOrUndef() || !EVal.getAs<NonLoc>())
    return false;
  DefinedSVal DefinedEVal = EVal.castAs<DefinedSVal>();

  SValBuilder &Bldr = C.getSValBuilder();
  DefinedSVal V = Bldr.makeIntVal(0, false);

  SVal LT =
      Bldr.evalBinOp(State, BO_LT, DefinedEVal, V, Bldr.getConditionType());

  // Is E value greater than MaxVal?
  ConstraintManager &CM = C.getConstraintManager();
  ProgramStateRef StNegative, StPositive;
  std::tie(StNegative, StPositive) =
      CM.assumeDual(State, LT.castAs<DefinedSVal>());

  return StNegative && !StPositive;
}

bool ConversionChecker::isLossOfPrecision(const ImplicitCastExpr *Cast,
                                        CheckerContext &C) const {
  // Don't warn about explicit loss of precision.
  if (Cast->isEvaluatable(C.getASTContext()))
    return false;

  QualType CastType = Cast->getType();
  QualType SubType = Cast->IgnoreParenImpCasts()->getType();

  if (!CastType->isIntegerType() || !SubType->isIntegerType())
    return false;

  if (C.getASTContext().getIntWidth(CastType) >=
      C.getASTContext().getIntWidth(SubType))
    return false;

  unsigned W = C.getASTContext().getIntWidth(CastType);
  if (W == 1 || W >= 64U)
    return false;

  unsigned long long MaxVal = 1ULL << W;
  return isGreaterEqual(C, Cast->getSubExpr(), MaxVal);
}

bool ConversionChecker::isLossOfSign(const ImplicitCastExpr *Cast,
                                   CheckerContext &C) const {
  QualType CastType = Cast->getType();
  QualType SubType = Cast->IgnoreParenImpCasts()->getType();

  if (!CastType->isUnsignedIntegerType() || !SubType->isSignedIntegerType())
    return false;

  return isNegative(C, Cast->getSubExpr());
}

void ento::registerConversionChecker(CheckerManager &mgr) {
  mgr.registerChecker<ConversionChecker>();
}
