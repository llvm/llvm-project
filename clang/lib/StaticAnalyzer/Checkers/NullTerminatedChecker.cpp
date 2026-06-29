//===- NullTerminatedChecker.cpp - Check null_terminated params -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This defines NullTerminatedChecker, which checks for arguments treated as
// buffers that are expected to be null-terminated (ends with zero-valued
// element). For constant-size arrays, the checker scans all elements and
// considers the array null-terminated if any element is constrained to zero.
//
// Parameters are marked as expecting null-terminated buffers using:
//   __attribute__((annotate("null_terminated")))
//
//===----------------------------------------------------------------------===//

#include "clang/AST/Attr.h"
#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"

using namespace clang;
using namespace ento;

namespace {
class NullTerminatedChecker : public Checker<check::PreCall> {
public:
  int MaxArraySize = 1024;

  void checkPreCall(const CallEvent &Call, CheckerContext &C) const;

private:
  const BugType BT{this, "Array not null-terminated", "API"};

  /// Return true if the parameter has annotate("null_terminated").
  static bool isNullTerminatedParam(const ParmVarDecl *Param);

  /// Return true if any element in [0, \param ArraySize) can be zero.
  bool hasAnyZeroElement(ProgramStateRef State, SValBuilder &SVB,
                         QualType EltTy, uint64_t ArraySize,
                         const TypedValueRegion *TVR) const;

  /// Return true if the element at \param Idx can be zero.
  bool canBeZero(ProgramStateRef State, SValBuilder &SVB, QualType EltTy,
                 SVal Idx, const TypedValueRegion *TVR) const;
};
} // namespace

bool NullTerminatedChecker::isNullTerminatedParam(const ParmVarDecl *Param) {
  return llvm::any_of(Param->specific_attrs<AnnotateAttr>(),
                      [](const AnnotateAttr *Ann) {
                        return Ann->getAnnotation() == "null_terminated";
                      });
}

bool NullTerminatedChecker::canBeZero(ProgramStateRef State, SValBuilder &SVB,
                                      QualType EltTy, SVal Idx,
                                      const TypedValueRegion *TVR) const {
  // Resolve element at Idx to a defined SVal.
  auto IdxNL = Idx.getAs<NonLoc>();
  if (!IdxNL)
    return true;
  SVal EltAddr = State->getLValue(EltTy, *IdxNL, loc::MemRegionVal(TVR));
  auto EltLocOpt = EltAddr.getAs<Loc>();
  if (!EltLocOpt)
    return true;
  SVal Val = State->getSVal(*EltLocOpt); // Load from addr
  auto DV = Val.getAs<DefinedSVal>();
  if (!DV)
    return true;

  // Is the element possibly zero on this path?
  SVal EqZero = SVB.evalEQ(State, *DV, SVB.makeZeroVal(EltTy));
  auto EqZeroDV = EqZero.getAs<DefinedSVal>();
  if (!EqZeroDV)
    return true;
  ProgramStateRef Zero, NonZero;
  std::tie(Zero, NonZero) = State->assume(*EqZeroDV);
  return static_cast<bool>(Zero);
}

bool NullTerminatedChecker::hasAnyZeroElement(
    ProgramStateRef State, SValBuilder &SVB, QualType EltTy, uint64_t ArraySize,
    const TypedValueRegion *TVR) const {
  QualType IdxTy = SVB.getArrayIndexType();
  for (uint64_t I = 0; I < ArraySize; ++I) {
    SVal Idx = SVB.makeIntVal(I, IdxTy);
    if (canBeZero(State, SVB, EltTy, Idx, TVR))
      return true;
  }
  return false;
}

void NullTerminatedChecker::checkPreCall(const CallEvent &Call,
                                         CheckerContext &C) const {
  const auto *FD = dyn_cast_or_null<FunctionDecl>(Call.getDecl());
  if (!FD)
    return;

  ProgramStateRef State = C.getState();
  SValBuilder &SVB = C.getSValBuilder();
  ASTContext &Ctx = C.getASTContext();

  unsigned NumParams = FD->getNumParams();
  unsigned NumArgs = Call.getNumArgs();

  // The call to min handles the case when |NumParams| != |NumArgs|.
  for (unsigned I = 0, N = std::min(NumParams, NumArgs); I < N; ++I) {
    const ParmVarDecl *Param = FD->getParamDecl(I);
    if (!isNullTerminatedParam(Param))
      continue;

    SVal ArgVal = Call.getArgSVal(I);
    const MemRegion *R = ArgVal.getAsRegion();
    if (!R)
      continue;

    // Strip ElementRegion wrappers (array-to-pointer decay produces
    // &Element{Array, 0}).
    R = R->StripCasts();
    if (const auto *ER = dyn_cast<ElementRegion>(R))
      R = ER->getSuperRegion();

    const auto *TVR = dyn_cast<TypedValueRegion>(R);
    if (!TVR)
      continue;

    QualType ElemTy;
    bool HasNullTerm = false;

    // Constant-size array (skips C99 FAMs).
    if (const auto *CAT = Ctx.getAsConstantArrayType(TVR->getValueType())) {
      uint64_t ArraySize = CAT->getSize().getZExtValue();

      // C89-style FAMs are dynamically allocated to larger sizes than as
      // declared so the analyzer cannot reason about them. Skip zero-length
      // arrays altogether, and one-length arrays if they are at the end of a
      // struct.
      if (ArraySize == 0 || ArraySize > static_cast<uint64_t>(MaxArraySize))
        continue;
      if (ArraySize == 1)
        if (const auto *FR = dyn_cast<FieldRegion>(TVR))
          if (FR->getDecl()->getFieldIndex() ==
              FR->getDecl()->getParent()->getNumFields() - 1)
            continue;
      ElemTy = CAT->getElementType();
      HasNullTerm = hasAnyZeroElement(State, SVB, ElemTy, ArraySize, TVR);
    } else {
      continue;
    }

    // TODO: Handle VLAs.

    if (HasNullTerm)
      continue;

    // Only warn when all elements constrained to non-zero values.
    if (ExplodedNode *N = C.generateNonFatalErrorNode(State)) {
      SmallString<128> Msg;
      llvm::raw_svector_ostream OS(Msg);
      OS << "array argument is not null-terminated; parameter "
         << Param->getName() << " expects a null-terminated array";
      auto Report = std::make_unique<PathSensitiveBugReport>(BT, Msg, N);
      Report->addRange(Call.getArgSourceRange(I));
      if (const Expr *ArgE = Call.getArgExpr(I))
        bugreporter::trackExpressionValue(N, ArgE, *Report);
      C.emitReport(std::move(Report));
    }
  }
}

void ento::registerNullTerminatedChecker(CheckerManager &Mgr) {
  auto *Checker = Mgr.registerChecker<NullTerminatedChecker>();
  Checker->MaxArraySize =
      Mgr.getAnalyzerOptions().getCheckerIntegerOption(Checker, "MaxArraySize");
  if (Checker->MaxArraySize < 0) {
    Mgr.reportInvalidCheckerOptionValue(Checker, "MaxArraySize",
                                        "a non-negative value");
    Checker->MaxArraySize = 0;
  }
}

bool ento::shouldRegisterNullTerminatedChecker(const CheckerManager &Mgr) {
  return true;
}
