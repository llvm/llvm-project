//===--- CIRGenAtomic.cpp - Emit CIR for atomic operations ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the code for emitting atomic operations.
//
//===----------------------------------------------------------------------===//

#include "CIRGenFunction.h"
#include "clang/CIR/MissingFeatures.h"

using namespace clang;
using namespace clang::CIRGen;
using namespace cir;

namespace {
class AtomicInfo {
  CIRGenFunction &cgf;
  QualType atomicTy;
  QualType valueTy;
  uint64_t atomicSizeInBits = 0;
  uint64_t valueSizeInBits = 0;
  CharUnits atomicAlign;
  CharUnits valueAlign;
  TypeEvaluationKind evaluationKind = cir::TEK_Scalar;
  LValue lvalue;
  mlir::Location loc;

public:
  AtomicInfo(CIRGenFunction &cgf, LValue &lvalue, mlir::Location loc)
      : cgf(cgf), loc(loc) {
    assert(!lvalue.isGlobalReg());
    ASTContext &ctx = cgf.getContext();
    if (lvalue.isSimple()) {
      atomicTy = lvalue.getType();
      if (auto *ty = atomicTy->getAs<AtomicType>())
        valueTy = ty->getValueType();
      else
        valueTy = atomicTy;
      evaluationKind = cgf.getEvaluationKind(valueTy);

      TypeInfo valueTypeInfo = ctx.getTypeInfo(valueTy);
      TypeInfo atomicTypeInfo = ctx.getTypeInfo(atomicTy);
      uint64_t valueAlignInBits = valueTypeInfo.Align;
      uint64_t atomicAlignInBits = atomicTypeInfo.Align;
      valueSizeInBits = valueTypeInfo.Width;
      atomicSizeInBits = atomicTypeInfo.Width;
      assert(valueSizeInBits <= atomicSizeInBits);
      assert(valueAlignInBits <= atomicAlignInBits);

      atomicAlign = ctx.toCharUnitsFromBits(atomicAlignInBits);
      valueAlign = ctx.toCharUnitsFromBits(valueAlignInBits);
      if (lvalue.getAlignment().isZero())
        lvalue.setAlignment(atomicAlign);

      this->lvalue = lvalue;
    } else {
      assert(!cir::MissingFeatures::atomicInfo());
      cgf.cgm.errorNYI(loc, "AtomicInfo: non-simple lvalue");
    }

    assert(!cir::MissingFeatures::atomicUseLibCall());
  }

  QualType getValueType() const { return valueTy; }
  CharUnits getAtomicAlignment() const { return atomicAlign; }
  TypeEvaluationKind getEvaluationKind() const { return evaluationKind; }
  mlir::Value getAtomicPointer() const {
    if (lvalue.isSimple())
      return lvalue.getPointer();
    assert(!cir::MissingFeatures::atomicInfoGetAtomicPointer());
    return nullptr;
  }
  Address getAtomicAddress() const {
    mlir::Type elemTy;
    if (lvalue.isSimple()) {
      elemTy = lvalue.getAddress().getElementType();
    } else {
      assert(!cir::MissingFeatures::atomicInfoGetAtomicAddress());
      cgf.cgm.errorNYI(loc, "AtomicInfo::getAtomicAddress: non-simple lvalue");
    }
    return Address(getAtomicPointer(), elemTy, getAtomicAlignment());
  }

  /// Is the atomic size larger than the underlying value type?
  ///
  /// Note that the absence of padding does not mean that atomic
  /// objects are completely interchangeable with non-atomic
  /// objects: we might have promoted the alignment of a type
  /// without making it bigger.
  bool hasPadding() const { return (valueSizeInBits != atomicSizeInBits); }

  bool emitMemSetZeroIfNecessary() const;

  /// Copy an atomic r-value into atomic-layout memory.
  void emitCopyIntoMemory(RValue rvalue) const;

  /// Project an l-value down to the value field.
  LValue projectValue() const {
    assert(lvalue.isSimple());
    Address addr = getAtomicAddress();
    if (hasPadding()) {
      cgf.cgm.errorNYI(loc, "AtomicInfo::projectValue: padding");
    }

    assert(!cir::MissingFeatures::opTBAA());
    return LValue::makeAddr(addr, getValueType(), lvalue.getBaseInfo());
  }

private:
  bool requiresMemSetZero(mlir::Type ty) const;
};
} // namespace

/// Does a store of the given IR type modify the full expected width?
static bool isFullSizeType(CIRGenModule &cgm, mlir::Type ty,
                           uint64_t expectedSize) {
  return cgm.getDataLayout().getTypeStoreSize(ty) * 8 == expectedSize;
}

/// Does the atomic type require memsetting to zero before initialization?
///
/// The IR type is provided as a way of making certain queries faster.
bool AtomicInfo::requiresMemSetZero(mlir::Type ty) const {
  // If the atomic type has size padding, we definitely need a memset.
  if (hasPadding())
    return true;

  // Otherwise, do some simple heuristics to try to avoid it:
  switch (getEvaluationKind()) {
  // For scalars and complexes, check whether the store size of the
  // type uses the full size.
  case cir::TEK_Scalar:
    return !isFullSizeType(cgf.cgm, ty, atomicSizeInBits);
  case cir::TEK_Complex:
    cgf.cgm.errorNYI(loc, "AtomicInfo::requiresMemSetZero: complex type");
    return false;

  // Padding in structs has an undefined bit pattern.  User beware.
  case cir::TEK_Aggregate:
    return false;
  }
  llvm_unreachable("bad evaluation kind");
}

bool AtomicInfo::emitMemSetZeroIfNecessary() const {
  assert(lvalue.isSimple());
  Address addr = lvalue.getAddress();
  if (!requiresMemSetZero(addr.getElementType()))
    return false;

  cgf.cgm.errorNYI(loc,
                   "AtomicInfo::emitMemSetZeroIfNecessary: emit memset zero");
  return false;
}

/// Copy an r-value into memory as part of storing to an atomic type.
/// This needs to create a bit-pattern suitable for atomic operations.
void AtomicInfo::emitCopyIntoMemory(RValue rvalue) const {
  assert(lvalue.isSimple());

  // If we have an r-value, the rvalue should be of the atomic type,
  // which means that the caller is responsible for having zeroed
  // any padding.  Just do an aggregate copy of that type.
  if (rvalue.isAggregate()) {
    cgf.cgm.errorNYI("copying aggregate into atomic lvalue");
    return;
  }

  // Okay, otherwise we're copying stuff.

  // Zero out the buffer if necessary.
  emitMemSetZeroIfNecessary();

  // Drill past the padding if present.
  LValue tempLValue = projectValue();

  // Okay, store the rvalue in.
  if (rvalue.isScalar()) {
    cgf.emitStoreOfScalar(rvalue.getValue(), tempLValue, /*isInit=*/true);
  } else {
    cgf.cgm.errorNYI("copying complex into atomic lvalue");
  }
}

RValue CIRGenFunction::emitAtomicExpr(AtomicExpr *e) {
  QualType atomicTy = e->getPtr()->getType()->getPointeeType();
  QualType memTy = atomicTy;
  if (const auto *ty = atomicTy->getAs<AtomicType>())
    memTy = ty->getValueType();

  Address ptr = emitPointerWithAlignment(e->getPtr());

  assert(!cir::MissingFeatures::openCL());
  if (e->getOp() == AtomicExpr::AO__c11_atomic_init) {
    LValue lvalue = makeAddrLValue(ptr, atomicTy);
    emitAtomicInit(e->getVal1(), lvalue);
    return RValue::get(nullptr);
  }

  assert(!cir::MissingFeatures::atomicExpr());
  cgm.errorNYI(e->getSourceRange(), "atomic expr is NYI");
  return RValue::get(nullptr);
}

void CIRGenFunction::emitAtomicInit(Expr *init, LValue dest) {
  AtomicInfo atomics(*this, dest, getLoc(init->getSourceRange()));

  switch (atomics.getEvaluationKind()) {
  case cir::TEK_Scalar: {
    mlir::Value value = emitScalarExpr(init);
    atomics.emitCopyIntoMemory(RValue::get(value));
    return;
  }

  case cir::TEK_Complex:
    cgm.errorNYI(init->getSourceRange(), "emitAtomicInit: complex type");
    return;

  case cir::TEK_Aggregate:
    cgm.errorNYI(init->getSourceRange(), "emitAtomicInit: aggregate type");
    return;
  }

  llvm_unreachable("bad evaluation kind");
}
