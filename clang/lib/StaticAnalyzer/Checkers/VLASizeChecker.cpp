//=== VLASizeChecker.cpp - Undefined dereference checker --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This defines VLASizeChecker, a builtin check in ExprEngine that
// performs checks for declaration of VLA of undefined or zero size.
// In addition, VLASizeChecker is responsible for defining the extent
// of the MemRegion that represents a VLA.
//
//===----------------------------------------------------------------------===//

#include "Taint.h"
#include "clang/AST/CharUnits.h"
#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/DynamicSize.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace ento;
using namespace taint;

namespace {
class VLASizeChecker : public Checker< check::PreStmt<DeclStmt> > {
  mutable std::unique_ptr<BugType> BT;
  enum VLASize_Kind { VLA_Garbage, VLA_Zero, VLA_Tainted, VLA_Negative };

  ProgramStateRef checkVLASize(CheckerContext &C, ProgramStateRef State,
                               const Expr *SizeE) const;

  void reportBug(VLASize_Kind Kind, const Expr *SizeE, ProgramStateRef State,
                 CheckerContext &C,
                 std::unique_ptr<BugReporterVisitor> Visitor = nullptr) const;

public:
  void checkPreStmt(const DeclStmt *DS, CheckerContext &C) const;
};
} // end anonymous namespace

ProgramStateRef VLASizeChecker::checkVLASize(CheckerContext &C,
                                             ProgramStateRef State,
                                             const Expr *SizeE) const {
  SVal SizeV = C.getSVal(SizeE);

  if (SizeV.isUndef()) {
    reportBug(VLA_Garbage, SizeE, State, C);
    return nullptr;
  }

  // See if the size value is known. It can't be undefined because we would have
  // warned about that already.
  if (SizeV.isUnknown())
    return nullptr;

  // Check if the size is tainted.
  if (isTainted(State, SizeV)) {
    reportBug(VLA_Tainted, SizeE, nullptr, C,
              std::make_unique<TaintBugVisitor>(SizeV));
    return nullptr;
  }

  // Check if the size is zero.
  DefinedSVal SizeD = SizeV.castAs<DefinedSVal>();

  ProgramStateRef StateNotZero, StateZero;
  std::tie(StateNotZero, StateZero) = State->assume(SizeD);

  if (StateZero && !StateNotZero) {
    reportBug(VLA_Zero, SizeE, StateZero, C);
    return nullptr;
  }

  // From this point on, assume that the size is not zero.
  State = StateNotZero;

  // Check if the size is negative.
  SValBuilder &SVB = C.getSValBuilder();

  QualType SizeTy = SizeE->getType();
  DefinedOrUnknownSVal Zero = SVB.makeZeroVal(SizeTy);

  SVal LessThanZeroVal = SVB.evalBinOp(State, BO_LT, SizeD, Zero, SizeTy);
  if (Optional<DefinedSVal> LessThanZeroDVal =
          LessThanZeroVal.getAs<DefinedSVal>()) {
    ConstraintManager &CM = C.getConstraintManager();
    ProgramStateRef StatePos, StateNeg;

    std::tie(StateNeg, StatePos) = CM.assumeDual(State, *LessThanZeroDVal);
    if (StateNeg && !StatePos) {
      reportBug(VLA_Negative, SizeE, State, C); // FIXME: StateNeg ?
      return nullptr;
    }
    State = StatePos;
  }

  return State;
}

void VLASizeChecker::reportBug(
    VLASize_Kind Kind, const Expr *SizeE, ProgramStateRef State,
    CheckerContext &C, std::unique_ptr<BugReporterVisitor> Visitor) const {
  // Generate an error node.
  ExplodedNode *N = C.generateErrorNode(State);
  if (!N)
    return;

  if (!BT)
    BT.reset(new BuiltinBug(
        this, "Dangerous variable-length array (VLA) declaration"));

  SmallString<256> buf;
  llvm::raw_svector_ostream os(buf);
  os << "Declared variable-length array (VLA) ";
  switch (Kind) {
  case VLA_Garbage:
    os << "uses a garbage value as its size";
    break;
  case VLA_Zero:
    os << "has zero size";
    break;
  case VLA_Tainted:
    os << "has tainted size";
    break;
  case VLA_Negative:
    os << "has negative size";
    break;
  }

  auto report = std::make_unique<PathSensitiveBugReport>(*BT, os.str(), N);
  report->addVisitor(std::move(Visitor));
  report->addRange(SizeE->getSourceRange());
  bugreporter::trackExpressionValue(N, SizeE, *report);
  C.emitReport(std::move(report));
}

void VLASizeChecker::checkPreStmt(const DeclStmt *DS, CheckerContext &C) const {
  if (!DS->isSingleDecl())
    return;

  const VarDecl *VD = dyn_cast<VarDecl>(DS->getSingleDecl());
  if (!VD)
    return;

  ASTContext &Ctx = C.getASTContext();
  SValBuilder &SVB = C.getSValBuilder();
  ProgramStateRef State = C.getState();

  const VariableArrayType *VLA = Ctx.getAsVariableArrayType(VD->getType());
  if (!VLA)
    return;

  llvm::SmallVector<const Expr *, 2> VLASizes;
  const VariableArrayType *VLALast = nullptr;
  // Walk over the VLAs for every dimension until a non-VLA is found.
  // Collect the sizes in VLASizes, put the most inner VLA to `VLALast`.
  // In "vla[x][2][y][3]" this will be the array for index "y".
  // There is a VariableArrayType for every dimension (here "x", "2", "y")
  // until a non-vla is found.
  while (VLA) {
    const Expr *SizeE = VLA->getSizeExpr();
    State = checkVLASize(C, State, SizeE);
    if (!State)
      return;
    VLASizes.push_back(SizeE);
    VLALast = VLA;
    VLA = Ctx.getAsVariableArrayType(VLA->getElementType());
  };
  assert(VLALast &&
         "Array should have at least one variably-modified dimension.");

  // VLASizeChecker is responsible for defining the extent of the array being
  // declared. We do this by multiplying the array length by the element size,
  // then matching that with the array region's extent symbol.

  CanQualType SizeTy = Ctx.getSizeType();
  // Get the element size.
  CharUnits EleSize = Ctx.getTypeSizeInChars(VLALast->getElementType());
  NonLoc ArraySize =
      SVB.makeIntVal(EleSize.getQuantity(), SizeTy).castAs<NonLoc>();

  for (const Expr *SizeE : VLASizes) {
    auto SizeD = C.getSVal(SizeE).castAs<DefinedSVal>();
    // Convert the array length to size_t.
    NonLoc IndexLength =
        SVB.evalCast(SizeD, SizeTy, SizeE->getType()).castAs<NonLoc>();
    // Multiply the array length by the element size.
    SVal Mul = SVB.evalBinOpNN(State, BO_Mul, ArraySize, IndexLength, SizeTy);
    if (auto MulNonLoc = Mul.getAs<NonLoc>()) {
      ArraySize = *MulNonLoc;
    } else {
      // Extent could not be determined.
      // The state was probably still updated by the validation checks.
      C.addTransition(State);
      return;
    }
  }

  // Finally, assume that the array's size matches the given size.
  const LocationContext *LC = C.getLocationContext();
  DefinedOrUnknownSVal DynSize =
      getDynamicSize(State, State->getRegion(VD, LC), SVB);

  DefinedOrUnknownSVal SizeIsKnown = SVB.evalEQ(State, DynSize, ArraySize);
  State = State->assume(SizeIsKnown, true);

  // Assume should not fail at this point.
  assert(State);

  // Remember our assumptions!
  C.addTransition(State);
}

void ento::registerVLASizeChecker(CheckerManager &mgr) {
  mgr.registerChecker<VLASizeChecker>();
}

bool ento::shouldRegisterVLASizeChecker(const CheckerManager &mgr) {
  return true;
}
