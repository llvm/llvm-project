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

  /// Cast the given pointer to an integer pointer suitable for atomic
  /// operations on the source.
  Address castToAtomicIntPointer(Address addr) const;

  /// If addr is compatible with the iN that will be used for an atomic
  /// operation, bitcast it. Otherwise, create a temporary that is suitable and
  /// copy the value across.
  Address convertToAtomicIntPointer(Address addr) const;

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

  /// Creates temp alloca for intermediate operations on atomic value.
  Address createTempAlloca() const;

private:
  bool requiresMemSetZero(mlir::Type ty) const;
};
} // namespace

// This function emits any expression (scalar, complex, or aggregate)
// into a temporary alloca.
static Address emitValToTemp(CIRGenFunction &cgf, Expr *e) {
  Address declPtr = cgf.createMemTemp(
      e->getType(), cgf.getLoc(e->getSourceRange()), ".atomictmp");
  cgf.emitAnyExprToMem(e, declPtr, e->getType().getQualifiers(),
                       /*Init*/ true);
  return declPtr;
}

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

Address AtomicInfo::convertToAtomicIntPointer(Address addr) const {
  mlir::Type ty = addr.getElementType();
  uint64_t sourceSizeInBits = cgf.cgm.getDataLayout().getTypeSizeInBits(ty);
  if (sourceSizeInBits != atomicSizeInBits) {
    cgf.cgm.errorNYI(
        loc,
        "AtomicInfo::convertToAtomicIntPointer: convert through temp alloca");
  }

  return castToAtomicIntPointer(addr);
}

Address AtomicInfo::createTempAlloca() const {
  Address tempAlloca = cgf.createMemTemp(
      (lvalue.isBitField() && valueSizeInBits > atomicSizeInBits) ? valueTy
                                                                  : atomicTy,
      getAtomicAlignment(), loc, "atomic-temp");

  // Cast to pointer to value type for bitfields.
  if (lvalue.isBitField()) {
    cgf.cgm.errorNYI(loc, "AtomicInfo::createTempAlloca: bitfield lvalue");
  }

  return tempAlloca;
}

Address AtomicInfo::castToAtomicIntPointer(Address addr) const {
  auto intTy = mlir::dyn_cast<cir::IntType>(addr.getElementType());
  // Don't bother with int casts if the integer size is the same.
  if (intTy && intTy.getWidth() == atomicSizeInBits)
    return addr;
  auto ty = cgf.getBuilder().getUIntNTy(atomicSizeInBits);
  return addr.withElementType(cgf.getBuilder(), ty);
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

static void emitAtomicOp(CIRGenFunction &cgf, AtomicExpr *expr, Address dest,
                         Address ptr, Address val1, uint64_t size,
                         cir::MemOrder order) {
  std::unique_ptr<AtomicScopeModel> scopeModel = expr->getScopeModel();
  if (scopeModel) {
    assert(!cir::MissingFeatures::atomicScope());
    cgf.cgm.errorNYI(expr->getSourceRange(), "emitAtomicOp: atomic scope");
    return;
  }

  assert(!cir::MissingFeatures::atomicSyncScopeID());

  CIRGenBuilderTy &builder = cgf.getBuilder();
  mlir::Location loc = cgf.getLoc(expr->getSourceRange());
  auto orderAttr = cir::MemOrderAttr::get(builder.getContext(), order);

  switch (expr->getOp()) {
  case AtomicExpr::AO__c11_atomic_init:
    llvm_unreachable("already handled!");

  case AtomicExpr::AO__c11_atomic_load:
  case AtomicExpr::AO__atomic_load_n:
  case AtomicExpr::AO__atomic_load: {
    cir::LoadOp load =
        builder.createLoad(loc, ptr, /*isVolatile=*/expr->isVolatile());

    assert(!cir::MissingFeatures::atomicSyncScopeID());

    load->setAttr("mem_order", orderAttr);

    builder.createStore(loc, load->getResult(0), dest);
    return;
  }

  case AtomicExpr::AO__c11_atomic_store:
  case AtomicExpr::AO__atomic_store_n:
  case AtomicExpr::AO__atomic_store: {
    cir::LoadOp loadVal1 = builder.createLoad(loc, val1);

    assert(!cir::MissingFeatures::atomicSyncScopeID());

    builder.createStore(loc, loadVal1, ptr, expr->isVolatile(),
                        /*align=*/mlir::IntegerAttr{}, orderAttr);
    return;
  }

  case AtomicExpr::AO__opencl_atomic_init:

  case AtomicExpr::AO__c11_atomic_compare_exchange_strong:
  case AtomicExpr::AO__hip_atomic_compare_exchange_strong:
  case AtomicExpr::AO__opencl_atomic_compare_exchange_strong:

  case AtomicExpr::AO__c11_atomic_compare_exchange_weak:
  case AtomicExpr::AO__opencl_atomic_compare_exchange_weak:
  case AtomicExpr::AO__hip_atomic_compare_exchange_weak:

  case AtomicExpr::AO__atomic_compare_exchange:
  case AtomicExpr::AO__atomic_compare_exchange_n:
  case AtomicExpr::AO__scoped_atomic_compare_exchange:
  case AtomicExpr::AO__scoped_atomic_compare_exchange_n:

  case AtomicExpr::AO__opencl_atomic_load:
  case AtomicExpr::AO__hip_atomic_load:
  case AtomicExpr::AO__scoped_atomic_load_n:
  case AtomicExpr::AO__scoped_atomic_load:

  case AtomicExpr::AO__opencl_atomic_store:
  case AtomicExpr::AO__hip_atomic_store:
  case AtomicExpr::AO__scoped_atomic_store:
  case AtomicExpr::AO__scoped_atomic_store_n:

  case AtomicExpr::AO__c11_atomic_exchange:
  case AtomicExpr::AO__hip_atomic_exchange:
  case AtomicExpr::AO__opencl_atomic_exchange:
  case AtomicExpr::AO__atomic_exchange_n:
  case AtomicExpr::AO__atomic_exchange:
  case AtomicExpr::AO__scoped_atomic_exchange_n:
  case AtomicExpr::AO__scoped_atomic_exchange:

  case AtomicExpr::AO__atomic_add_fetch:
  case AtomicExpr::AO__scoped_atomic_add_fetch:

  case AtomicExpr::AO__c11_atomic_fetch_add:
  case AtomicExpr::AO__hip_atomic_fetch_add:
  case AtomicExpr::AO__opencl_atomic_fetch_add:
  case AtomicExpr::AO__atomic_fetch_add:
  case AtomicExpr::AO__scoped_atomic_fetch_add:

  case AtomicExpr::AO__atomic_sub_fetch:
  case AtomicExpr::AO__scoped_atomic_sub_fetch:

  case AtomicExpr::AO__c11_atomic_fetch_sub:
  case AtomicExpr::AO__hip_atomic_fetch_sub:
  case AtomicExpr::AO__opencl_atomic_fetch_sub:
  case AtomicExpr::AO__atomic_fetch_sub:
  case AtomicExpr::AO__scoped_atomic_fetch_sub:

  case AtomicExpr::AO__atomic_min_fetch:
  case AtomicExpr::AO__scoped_atomic_min_fetch:

  case AtomicExpr::AO__c11_atomic_fetch_min:
  case AtomicExpr::AO__hip_atomic_fetch_min:
  case AtomicExpr::AO__opencl_atomic_fetch_min:
  case AtomicExpr::AO__atomic_fetch_min:
  case AtomicExpr::AO__scoped_atomic_fetch_min:

  case AtomicExpr::AO__atomic_max_fetch:
  case AtomicExpr::AO__scoped_atomic_max_fetch:

  case AtomicExpr::AO__c11_atomic_fetch_max:
  case AtomicExpr::AO__hip_atomic_fetch_max:
  case AtomicExpr::AO__opencl_atomic_fetch_max:
  case AtomicExpr::AO__atomic_fetch_max:
  case AtomicExpr::AO__scoped_atomic_fetch_max:

  case AtomicExpr::AO__atomic_and_fetch:
  case AtomicExpr::AO__scoped_atomic_and_fetch:

  case AtomicExpr::AO__c11_atomic_fetch_and:
  case AtomicExpr::AO__hip_atomic_fetch_and:
  case AtomicExpr::AO__opencl_atomic_fetch_and:
  case AtomicExpr::AO__atomic_fetch_and:
  case AtomicExpr::AO__scoped_atomic_fetch_and:

  case AtomicExpr::AO__atomic_or_fetch:
  case AtomicExpr::AO__scoped_atomic_or_fetch:

  case AtomicExpr::AO__c11_atomic_fetch_or:
  case AtomicExpr::AO__hip_atomic_fetch_or:
  case AtomicExpr::AO__opencl_atomic_fetch_or:
  case AtomicExpr::AO__atomic_fetch_or:
  case AtomicExpr::AO__scoped_atomic_fetch_or:

  case AtomicExpr::AO__atomic_xor_fetch:
  case AtomicExpr::AO__scoped_atomic_xor_fetch:

  case AtomicExpr::AO__c11_atomic_fetch_xor:
  case AtomicExpr::AO__hip_atomic_fetch_xor:
  case AtomicExpr::AO__opencl_atomic_fetch_xor:
  case AtomicExpr::AO__atomic_fetch_xor:
  case AtomicExpr::AO__scoped_atomic_fetch_xor:

  case AtomicExpr::AO__atomic_nand_fetch:
  case AtomicExpr::AO__scoped_atomic_nand_fetch:

  case AtomicExpr::AO__c11_atomic_fetch_nand:
  case AtomicExpr::AO__atomic_fetch_nand:
  case AtomicExpr::AO__scoped_atomic_fetch_nand:

  case AtomicExpr::AO__atomic_test_and_set:

  case AtomicExpr::AO__atomic_clear:
    cgf.cgm.errorNYI(expr->getSourceRange(), "emitAtomicOp: expr op NYI");
    break;
  }
}

static bool isMemOrderValid(uint64_t order, bool isStore, bool isLoad) {
  if (!cir::isValidCIRAtomicOrderingCABI(order))
    return false;
  auto memOrder = static_cast<cir::MemOrder>(order);
  if (isStore)
    return memOrder != cir::MemOrder::Consume &&
           memOrder != cir::MemOrder::Acquire &&
           memOrder != cir::MemOrder::AcquireRelease;
  if (isLoad)
    return memOrder != cir::MemOrder::Release &&
           memOrder != cir::MemOrder::AcquireRelease;
  return true;
}

RValue CIRGenFunction::emitAtomicExpr(AtomicExpr *e) {
  QualType atomicTy = e->getPtr()->getType()->getPointeeType();
  QualType memTy = atomicTy;
  if (const auto *ty = atomicTy->getAs<AtomicType>())
    memTy = ty->getValueType();

  Address val1 = Address::invalid();
  Address dest = Address::invalid();
  Address ptr = emitPointerWithAlignment(e->getPtr());

  assert(!cir::MissingFeatures::openCL());
  if (e->getOp() == AtomicExpr::AO__c11_atomic_init) {
    LValue lvalue = makeAddrLValue(ptr, atomicTy);
    emitAtomicInit(e->getVal1(), lvalue);
    return RValue::get(nullptr);
  }

  TypeInfoChars typeInfo = getContext().getTypeInfoInChars(atomicTy);
  uint64_t size = typeInfo.Width.getQuantity();

  Expr::EvalResult orderConst;
  mlir::Value order;
  if (!e->getOrder()->EvaluateAsInt(orderConst, getContext()))
    order = emitScalarExpr(e->getOrder());

  bool shouldCastToIntPtrTy = true;

  switch (e->getOp()) {
  default:
    cgm.errorNYI(e->getSourceRange(), "atomic op NYI");
    return RValue::get(nullptr);

  case AtomicExpr::AO__c11_atomic_init:
    llvm_unreachable("already handled above with emitAtomicInit");

  case AtomicExpr::AO__atomic_load_n:
  case AtomicExpr::AO__c11_atomic_load:
    break;

  case AtomicExpr::AO__atomic_load:
    dest = emitPointerWithAlignment(e->getVal1());
    break;

  case AtomicExpr::AO__atomic_store:
    val1 = emitPointerWithAlignment(e->getVal1());
    break;

  case AtomicExpr::AO__atomic_store_n:
  case AtomicExpr::AO__c11_atomic_store:
    val1 = emitValToTemp(*this, e->getVal1());
    break;
  }

  QualType resultTy = e->getType().getUnqualifiedType();

  // The inlined atomics only function on iN types, where N is a power of 2. We
  // need to make sure (via temporaries if necessary) that all incoming values
  // are compatible.
  LValue atomicValue = makeAddrLValue(ptr, atomicTy);
  AtomicInfo atomics(*this, atomicValue, getLoc(e->getSourceRange()));

  if (shouldCastToIntPtrTy) {
    ptr = atomics.castToAtomicIntPointer(ptr);
    if (val1.isValid())
      val1 = atomics.convertToAtomicIntPointer(val1);
  }
  if (dest.isValid()) {
    if (shouldCastToIntPtrTy)
      dest = atomics.castToAtomicIntPointer(dest);
  } else if (!resultTy->isVoidType()) {
    dest = atomics.createTempAlloca();
    if (shouldCastToIntPtrTy)
      dest = atomics.castToAtomicIntPointer(dest);
  }

  bool powerOf2Size = (size & (size - 1)) == 0;
  bool useLibCall = !powerOf2Size || (size > 16);

  // For atomics larger than 16 bytes, emit a libcall from the frontend. This
  // avoids the overhead of dealing with excessively-large value types in IR.
  // Non-power-of-2 values also lower to libcall here, as they are not currently
  // permitted in IR instructions (although that constraint could be relaxed in
  // the future). For other cases where a libcall is required on a given
  // platform, we let the backend handle it (this includes handling for all of
  // the size-optimized libcall variants, which are only valid up to 16 bytes.)
  //
  // See: https://llvm.org/docs/Atomics.html#libcalls-atomic
  if (useLibCall) {
    assert(!cir::MissingFeatures::atomicUseLibCall());
    cgm.errorNYI(e->getSourceRange(), "emitAtomicExpr: emit atomic lib call");
    return RValue::get(nullptr);
  }

  bool isStore = e->getOp() == AtomicExpr::AO__c11_atomic_store ||
                 e->getOp() == AtomicExpr::AO__opencl_atomic_store ||
                 e->getOp() == AtomicExpr::AO__hip_atomic_store ||
                 e->getOp() == AtomicExpr::AO__atomic_store ||
                 e->getOp() == AtomicExpr::AO__atomic_store_n ||
                 e->getOp() == AtomicExpr::AO__scoped_atomic_store ||
                 e->getOp() == AtomicExpr::AO__scoped_atomic_store_n ||
                 e->getOp() == AtomicExpr::AO__atomic_clear;
  bool isLoad = e->getOp() == AtomicExpr::AO__c11_atomic_load ||
                e->getOp() == AtomicExpr::AO__opencl_atomic_load ||
                e->getOp() == AtomicExpr::AO__hip_atomic_load ||
                e->getOp() == AtomicExpr::AO__atomic_load ||
                e->getOp() == AtomicExpr::AO__atomic_load_n ||
                e->getOp() == AtomicExpr::AO__scoped_atomic_load ||
                e->getOp() == AtomicExpr::AO__scoped_atomic_load_n;

  if (!order) {
    // We have evaluated the memory order as an integer constant in orderConst.
    // We should not ever get to a case where the ordering isn't a valid CABI
    // value, but it's hard to enforce that in general.
    uint64_t ord = orderConst.Val.getInt().getZExtValue();
    if (isMemOrderValid(ord, isStore, isLoad))
      emitAtomicOp(*this, e, dest, ptr, val1, size,
                   static_cast<cir::MemOrder>(ord));
  } else {
    assert(!cir::MissingFeatures::atomicExpr());
    cgm.errorNYI(e->getSourceRange(), "emitAtomicExpr: dynamic memory order");
    return RValue::get(nullptr);
  }

  if (resultTy->isVoidType())
    return RValue::get(nullptr);

  return convertTempToRValue(
      dest.withElementType(builder, convertTypeForMem(resultTy)), resultTy,
      e->getExprLoc());
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
