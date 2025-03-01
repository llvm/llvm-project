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

#include "Address.h"

#include "CIRGenFunction.h"
#include "CIRGenModule.h"
#include "CIRGenOpenMPRuntime.h"
#include "TargetInfo.h"
#include "clang/AST/ASTContext.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDataLayout.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/MissingFeatures.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstdint>

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"

using namespace clang;
using namespace clang::CIRGen;

namespace {
class AtomicInfo {
  CIRGenFunction &CGF;
  QualType AtomicTy;
  QualType ValueTy;
  uint64_t AtomicSizeInBits;
  uint64_t ValueSizeInBits;
  CharUnits AtomicAlign;
  CharUnits ValueAlign;
  cir::TypeEvaluationKind EvaluationKind;
  bool UseLibcall;
  LValue LVal;
  CIRGenBitFieldInfo BFI;
  mlir::Location loc;

public:
  AtomicInfo(CIRGenFunction &CGF, LValue &lvalue, mlir::Location l)
      : CGF(CGF), AtomicSizeInBits(0), ValueSizeInBits(0),
        EvaluationKind(cir::TEK_Scalar), UseLibcall(true), loc(l) {
    assert(!lvalue.isGlobalReg());
    ASTContext &C = CGF.getContext();
    if (lvalue.isSimple()) {
      AtomicTy = lvalue.getType();
      if (auto *ATy = AtomicTy->getAs<AtomicType>())
        ValueTy = ATy->getValueType();
      else
        ValueTy = AtomicTy;
      EvaluationKind = CGF.getEvaluationKind(ValueTy);

      uint64_t ValueAlignInBits;
      uint64_t AtomicAlignInBits;
      TypeInfo ValueTI = C.getTypeInfo(ValueTy);
      ValueSizeInBits = ValueTI.Width;
      ValueAlignInBits = ValueTI.Align;

      TypeInfo AtomicTI = C.getTypeInfo(AtomicTy);
      AtomicSizeInBits = AtomicTI.Width;
      AtomicAlignInBits = AtomicTI.Align;

      assert(ValueSizeInBits <= AtomicSizeInBits);
      assert(ValueAlignInBits <= AtomicAlignInBits);

      AtomicAlign = C.toCharUnitsFromBits(AtomicAlignInBits);
      ValueAlign = C.toCharUnitsFromBits(ValueAlignInBits);
      if (lvalue.getAlignment().isZero())
        lvalue.setAlignment(AtomicAlign);

      LVal = lvalue;
    } else if (lvalue.isBitField()) {
      llvm_unreachable("NYI");
    } else if (lvalue.isVectorElt()) {
      ValueTy = lvalue.getType()->castAs<VectorType>()->getElementType();
      ValueSizeInBits = C.getTypeSize(ValueTy);
      AtomicTy = lvalue.getType();
      AtomicSizeInBits = C.getTypeSize(AtomicTy);
      AtomicAlign = ValueAlign = lvalue.getAlignment();
      LVal = lvalue;
    } else {
      llvm_unreachable("NYI");
    }
    UseLibcall = !C.getTargetInfo().hasBuiltinAtomic(
        AtomicSizeInBits, C.toBits(lvalue.getAlignment()));
  }

  QualType getAtomicType() const { return AtomicTy; }
  QualType getValueType() const { return ValueTy; }
  CharUnits getAtomicAlignment() const { return AtomicAlign; }
  uint64_t getAtomicSizeInBits() const { return AtomicSizeInBits; }
  uint64_t getValueSizeInBits() const { return ValueSizeInBits; }
  cir::TypeEvaluationKind getEvaluationKind() const { return EvaluationKind; }
  bool shouldUseLibcall() const { return UseLibcall; }
  const LValue &getAtomicLValue() const { return LVal; }
  mlir::Value getAtomicPointer() const {
    if (LVal.isSimple())
      return LVal.getPointer();
    else if (LVal.isBitField())
      return LVal.getBitFieldPointer();
    else if (LVal.isVectorElt())
      return LVal.getVectorPointer();
    assert(LVal.isExtVectorElt());
    // TODO(cir): return LVal.getExtVectorPointer();
    llvm_unreachable("NYI");
  }
  Address getAtomicAddress() const {
    mlir::Type ElTy;
    if (LVal.isSimple())
      ElTy = LVal.getAddress().getElementType();
    else if (LVal.isBitField())
      ElTy = LVal.getBitFieldAddress().getElementType();
    else if (LVal.isVectorElt())
      ElTy = LVal.getVectorAddress().getElementType();
    else // TODO(cir): ElTy = LVal.getExtVectorAddress().getElementType();
      llvm_unreachable("NYI");
    return Address(getAtomicPointer(), ElTy, getAtomicAlignment());
  }

  Address getAtomicAddressAsAtomicIntPointer() const {
    return castToAtomicIntPointer(getAtomicAddress());
  }

  /// Is the atomic size larger than the underlying value type?
  ///
  /// Note that the absence of padding does not mean that atomic
  /// objects are completely interchangeable with non-atomic
  /// objects: we might have promoted the alignment of a type
  /// without making it bigger.
  bool hasPadding() const { return (ValueSizeInBits != AtomicSizeInBits); }

  bool emitMemSetZeroIfNecessary() const;

  mlir::Value getAtomicSizeValue() const { llvm_unreachable("NYI"); }

  mlir::Value getScalarRValValueOrNull(RValue RVal) const;

  /// Cast the given pointer to an integer pointer suitable for atomic
  /// operations if the source.
  Address castToAtomicIntPointer(Address Addr) const;

  /// If Addr is compatible with the iN that will be used for an atomic
  /// operation, bitcast it. Otherwise, create a temporary that is suitable
  /// and copy the value across.
  Address convertToAtomicIntPointer(Address Addr) const;

  /// Turn an atomic-layout object into an r-value.
  RValue convertAtomicTempToRValue(Address addr, AggValueSlot resultSlot,
                                   SourceLocation loc, bool AsValue) const;

  /// Converts a rvalue to integer value.
  mlir::Value convertRValueToInt(RValue RVal, bool CmpXchg = false) const;

  RValue ConvertIntToValueOrAtomic(mlir::Value IntVal, AggValueSlot ResultSlot,
                                   SourceLocation Loc, bool AsValue) const;

  /// Copy an atomic r-value into atomic-layout memory.
  void emitCopyIntoMemory(RValue rvalue) const;

  /// Project an l-value down to the value field.
  LValue projectValue() const {
    assert(LVal.isSimple());
    Address addr = getAtomicAddress();
    if (hasPadding())
      llvm_unreachable("NYI");

    return LValue::makeAddr(addr, getValueType(), CGF.getContext(),
                            LVal.getBaseInfo(), LVal.getTBAAInfo());
  }

  /// Emits atomic load.
  /// \returns Loaded value.
  RValue EmitAtomicLoad(AggValueSlot ResultSlot, SourceLocation Loc,
                        bool AsValue, llvm::AtomicOrdering AO, bool IsVolatile);

  /// Emits atomic compare-and-exchange sequence.
  /// \param Expected Expected value.
  /// \param Desired Desired value.
  /// \param Success Atomic ordering for success operation.
  /// \param Failure Atomic ordering for failed operation.
  /// \param IsWeak true if atomic operation is weak, false otherwise.
  /// \returns Pair of values: previous value from storage (value type) and
  /// boolean flag (i1 type) with true if success and false otherwise.
  std::pair<RValue, mlir::Value>
  EmitAtomicCompareExchange(RValue Expected, RValue Desired,
                            llvm::AtomicOrdering Success =
                                llvm::AtomicOrdering::SequentiallyConsistent,
                            llvm::AtomicOrdering Failure =
                                llvm::AtomicOrdering::SequentiallyConsistent,
                            bool IsWeak = false);

  /// Emits atomic update.
  /// \param AO Atomic ordering.
  /// \param UpdateOp Update operation for the current lvalue.
  void EmitAtomicUpdate(llvm::AtomicOrdering AO,
                        const llvm::function_ref<RValue(RValue)> &UpdateOp,
                        bool IsVolatile);
  /// Emits atomic update.
  /// \param AO Atomic ordering.
  void EmitAtomicUpdate(llvm::AtomicOrdering AO, RValue UpdateRVal,
                        bool IsVolatile);

  /// Materialize an atomic r-value in atomic-layout memory.
  Address materializeRValue(RValue rvalue) const;

  /// Creates temp alloca for intermediate operations on atomic value.
  Address CreateTempAlloca() const;

private:
  bool requiresMemSetZero(mlir::Type ty) const;

  /// Emits atomic load as a libcall.
  void EmitAtomicLoadLibcall(mlir::Value AddForLoaded, llvm::AtomicOrdering AO,
                             bool IsVolatile);
  /// Emits atomic load as LLVM instruction.
  mlir::Value EmitAtomicLoadOp(llvm::AtomicOrdering AO, bool IsVolatile);
  /// Emits atomic compare-and-exchange op as a libcall.
  mlir::Value EmitAtomicCompareExchangeLibcall(
      mlir::Value ExpectedAddr, mlir::Value DesiredAddr,
      llvm::AtomicOrdering Success =
          llvm::AtomicOrdering::SequentiallyConsistent,
      llvm::AtomicOrdering Failure =
          llvm::AtomicOrdering::SequentiallyConsistent);
  /// Emits atomic compare-and-exchange op as LLVM instruction.
  std::pair<mlir::Value, mlir::Value>
  EmitAtomicCompareExchangeOp(mlir::Value ExpectedVal, mlir::Value DesiredVal,
                              llvm::AtomicOrdering Success =
                                  llvm::AtomicOrdering::SequentiallyConsistent,
                              llvm::AtomicOrdering Failure =
                                  llvm::AtomicOrdering::SequentiallyConsistent,
                              bool IsWeak = false);
  /// Emit atomic update as libcalls.
  void
  EmitAtomicUpdateLibcall(llvm::AtomicOrdering AO,
                          const llvm::function_ref<RValue(RValue)> &UpdateOp,
                          bool IsVolatile);
  /// Emit atomic update as LLVM instructions.
  void EmitAtomicUpdateOp(llvm::AtomicOrdering AO,
                          const llvm::function_ref<RValue(RValue)> &UpdateOp,
                          bool IsVolatile);
  /// Emit atomic update as libcalls.
  void EmitAtomicUpdateLibcall(llvm::AtomicOrdering AO, RValue UpdateRVal,
                               bool IsVolatile);
  /// Emit atomic update as LLVM instructions.
  void EmitAtomicUpdateOp(llvm::AtomicOrdering AO, RValue UpdateRal,
                          bool IsVolatile);
};
} // namespace

// This function emits any expression (scalar, complex, or aggregate)
// into a temporary alloca.
static Address emitValToTemp(CIRGenFunction &CGF, Expr *E) {
  Address DeclPtr = CGF.CreateMemTemp(
      E->getType(), CGF.getLoc(E->getSourceRange()), ".atomictmp");
  CGF.emitAnyExprToMem(E, DeclPtr, E->getType().getQualifiers(),
                       /*Init*/ true);
  return DeclPtr;
}

/// Does a store of the given IR type modify the full expected width?
static bool isFullSizeType(CIRGenModule &CGM, mlir::Type ty,
                           uint64_t expectedSize) {
  return (CGM.getDataLayout().getTypeStoreSize(ty) * 8 == expectedSize);
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
    return !isFullSizeType(CGF.CGM, ty, AtomicSizeInBits);
  case cir::TEK_Complex:
    llvm_unreachable("NYI");

  // Padding in structs has an undefined bit pattern.  User beware.
  case cir::TEK_Aggregate:
    return false;
  }
  llvm_unreachable("bad evaluation kind");
}

Address AtomicInfo::castToAtomicIntPointer(Address addr) const {
  auto intTy = mlir::dyn_cast<cir::IntType>(addr.getElementType());
  // Don't bother with int casts if the integer size is the same.
  if (intTy && intTy.getWidth() == AtomicSizeInBits)
    return addr;
  auto ty = CGF.getBuilder().getUIntNTy(AtomicSizeInBits);
  return addr.withElementType(CGF.getBuilder(), ty);
}

Address AtomicInfo::convertToAtomicIntPointer(Address Addr) const {
  auto Ty = Addr.getElementType();
  uint64_t SourceSizeInBits = CGF.CGM.getDataLayout().getTypeSizeInBits(Ty);
  if (SourceSizeInBits != AtomicSizeInBits) {
    llvm_unreachable("NYI");
  }

  return castToAtomicIntPointer(Addr);
}

Address AtomicInfo::CreateTempAlloca() const {
  Address TempAlloca = CGF.CreateMemTemp(
      (LVal.isBitField() && ValueSizeInBits > AtomicSizeInBits) ? ValueTy
                                                                : AtomicTy,
      getAtomicAlignment(), loc, "atomic-temp");
  // Cast to pointer to value type for bitfields.
  if (LVal.isBitField()) {
    llvm_unreachable("NYI");
  }
  return TempAlloca;
}

// If the value comes from a ConstOp + IntAttr, retrieve and skip a series
// of casts if necessary.
//
// FIXME(cir): figure out warning issue and move this to CIRBaseBuilder.h
static cir::IntAttr getConstOpIntAttr(mlir::Value v) {
  mlir::Operation *op = v.getDefiningOp();
  cir::IntAttr constVal;
  while (auto c = dyn_cast<cir::CastOp>(op))
    op = c.getOperand().getDefiningOp();
  if (auto c = dyn_cast<cir::ConstantOp>(op)) {
    if (mlir::isa<cir::IntType>(c.getType()))
      constVal = mlir::cast<cir::IntAttr>(c.getValue());
  }
  return constVal;
}

// Inspect a value that is the strong/weak flag for a compare-exchange.  If it
// is a constant of intergral or boolean type, set `val` to the constant's
// boolean value and return true.  Otherwise leave `val` unchanged and return
// false.
static bool isCstWeak(mlir::Value weakVal, bool &val) {
  mlir::Operation *op = weakVal.getDefiningOp();
  while (auto c = dyn_cast<cir::CastOp>(op)) {
    op = c.getOperand().getDefiningOp();
  }
  if (auto c = dyn_cast<cir::ConstantOp>(op)) {
    if (mlir::isa<cir::IntType>(c.getType())) {
      val = mlir::cast<cir::IntAttr>(c.getValue()).getUInt() != 0;
      return true;
    } else if (mlir::isa<cir::BoolType>(c.getType())) {
      val = mlir::cast<cir::BoolAttr>(c.getValue()).getValue();
      return true;
    }
  }
  return false;
}

// Functions that help with the creation of compiler-generated switch
// statements that are used to implement non-constant memory order parameters.

// Create a "default:" label and add it to the given collection of case labels.
// Create the region that will hold the body of the "default:" block.
static void emitDefaultCase(CIRGenBuilderTy &builder, mlir::Location loc) {
  auto EmptyArrayAttr = builder.getArrayAttr({});
  mlir::OpBuilder::InsertPoint insertPoint;
  builder.create<cir::CaseOp>(loc, EmptyArrayAttr, cir::CaseOpKind::Default,
                              insertPoint);
  builder.restoreInsertionPoint(insertPoint);
}

// Create a single "case" label with the given MemOrder as its value.  Add the
// "case" label to the given collection of case labels.  Create the region that
// will hold the body of the "case" block.
static void emitSingleMemOrderCase(CIRGenBuilderTy &builder, mlir::Location loc,
                                   mlir::Type Type, cir::MemOrder Order) {
  SmallVector<mlir::Attribute, 1> OneOrder{
      cir::IntAttr::get(Type, static_cast<int>(Order))};
  auto OneAttribute = builder.getArrayAttr(OneOrder);
  mlir::OpBuilder::InsertPoint insertPoint;
  builder.create<cir::CaseOp>(loc, OneAttribute, cir::CaseOpKind::Equal,
                              insertPoint);
  builder.restoreInsertionPoint(insertPoint);
}

// Create a pair of "case" labels with the given MemOrders as their values.
// Add the combined "case" attribute to the given collection of case labels.
// Create the region that will hold the body of the "case" block.
static void emitDoubleMemOrderCase(CIRGenBuilderTy &builder, mlir::Location loc,
                                   mlir::Type Type, cir::MemOrder Order1,
                                   cir::MemOrder Order2) {
  SmallVector<mlir::Attribute, 2> TwoOrders{
      cir::IntAttr::get(Type, static_cast<int>(Order1)),
      cir::IntAttr::get(Type, static_cast<int>(Order2))};
  auto TwoAttributes = builder.getArrayAttr(TwoOrders);
  mlir::OpBuilder::InsertPoint insertPoint;
  builder.create<cir::CaseOp>(loc, TwoAttributes, cir::CaseOpKind::Anyof,
                              insertPoint);
  builder.restoreInsertionPoint(insertPoint);
}

static void emitAtomicCmpXchg(CIRGenFunction &CGF, AtomicExpr *E, bool IsWeak,
                              Address Dest, Address Ptr, Address Val1,
                              Address Val2, uint64_t Size,
                              cir::MemOrder SuccessOrder,
                              cir::MemOrder FailureOrder,
                              cir::MemScopeKind Scope) {
  auto &builder = CGF.getBuilder();
  auto loc = CGF.getLoc(E->getSourceRange());
  auto Expected = builder.createLoad(loc, Val1);
  auto Desired = builder.createLoad(loc, Val2);
  auto boolTy = builder.getBoolTy();
  auto cmpxchg = builder.create<cir::AtomicCmpXchg>(
      loc, Expected.getType(), boolTy, Ptr.getPointer(), Expected, Desired,
      cir::MemOrderAttr::get(&CGF.getMLIRContext(), SuccessOrder),
      cir::MemOrderAttr::get(&CGF.getMLIRContext(), FailureOrder),
      cir::MemScopeKindAttr::get(&CGF.getMLIRContext(), Scope),
      builder.getI64IntegerAttr(Ptr.getAlignment().getAsAlign().value()));
  cmpxchg.setIsVolatile(E->isVolatile());
  cmpxchg.setWeak(IsWeak);

  auto cmp = builder.createNot(cmpxchg.getCmp());
  builder.create<cir::IfOp>(
      loc, cmp, false, [&](mlir::OpBuilder &, mlir::Location) {
        auto ptrTy = mlir::cast<cir::PointerType>(Val1.getPointer().getType());
        if (Val1.getElementType() != ptrTy.getPointee()) {
          Val1 = Val1.withPointer(builder.createPtrBitcast(
              Val1.getPointer(), Val1.getElementType()));
        }
        builder.createStore(loc, cmpxchg.getOld(), Val1);
        builder.createYield(loc);
      });

  // Update the memory at Dest with Cmp's value.
  CGF.emitStoreOfScalar(cmpxchg.getCmp(),
                        CGF.makeAddrLValue(Dest, E->getType()));
}

/// Given an ordering required on success, emit all possible cmpxchg
/// instructions to cope with the provided (but possibly only dynamically known)
/// FailureOrder.
static void emitAtomicCmpXchgFailureSet(
    CIRGenFunction &CGF, AtomicExpr *E, bool IsWeak, Address Dest, Address Ptr,
    Address Val1, Address Val2, mlir::Value FailureOrderVal, uint64_t Size,
    cir::MemOrder SuccessOrder, cir::MemScopeKind Scope) {

  cir::MemOrder FailureOrder;
  if (auto ordAttr = getConstOpIntAttr(FailureOrderVal)) {
    // We should not ever get to a case where the ordering isn't a valid CABI
    // value, but it's hard to enforce that in general.
    auto ord = ordAttr.getUInt();
    if (!cir::isValidCIRAtomicOrderingCABI(ord)) {
      FailureOrder = cir::MemOrder::Relaxed;
    } else {
      switch ((cir::MemOrder)ord) {
      case cir::MemOrder::Relaxed:
        // 31.7.2.18: "The failure argument shall not be memory_order_release
        // nor memory_order_acq_rel". Fallback to monotonic.
      case cir::MemOrder::Release:
      case cir::MemOrder::AcquireRelease:
        FailureOrder = cir::MemOrder::Relaxed;
        break;
      case cir::MemOrder::Consume:
      case cir::MemOrder::Acquire:
        FailureOrder = cir::MemOrder::Acquire;
        break;
      case cir::MemOrder::SequentiallyConsistent:
        FailureOrder = cir::MemOrder::SequentiallyConsistent;
        break;
      }
    }
    // Prior to c++17, "the failure argument shall be no stronger than the
    // success argument". This condition has been lifted and the only
    // precondition is 31.7.2.18. Effectively treat this as a DR and skip
    // language version checks.
    emitAtomicCmpXchg(CGF, E, IsWeak, Dest, Ptr, Val1, Val2, Size, SuccessOrder,
                      FailureOrder, Scope);
    return;
  }

  // The failure memory order is not a compile-time value. The CIR atomic ops
  // can't handle a runtime value; all memory orders must be hard coded.
  // Generate a "switch" statement that converts the runtime value into a
  // compile-time value.
  CGF.getBuilder().create<cir::SwitchOp>(
      FailureOrderVal.getLoc(), FailureOrderVal,
      [&](mlir::OpBuilder &b, mlir::Location loc, mlir::OperationState &os) {
        auto &builder = CGF.getBuilder();

        mlir::Block *switchBlock = builder.getBlock();

        // default:
        // Unsupported memory orders get generated as memory_order_relaxed,
        // because there is no practical way to report an error at runtime.
        emitDefaultCase(builder, loc);
        emitAtomicCmpXchg(CGF, E, IsWeak, Dest, Ptr, Val1, Val2, Size,
                          SuccessOrder, cir::MemOrder::Relaxed, Scope);
        builder.createBreak(loc);

        builder.setInsertionPointToEnd(switchBlock);

        // case consume:
        // case acquire:
        // memory_order_consume is not implemented and always falls back to
        // memory_order_acquire
        emitDoubleMemOrderCase(builder, loc, FailureOrderVal.getType(),
                               cir::MemOrder::Consume, cir::MemOrder::Acquire);
        emitAtomicCmpXchg(CGF, E, IsWeak, Dest, Ptr, Val1, Val2, Size,
                          SuccessOrder, cir::MemOrder::Acquire, Scope);
        builder.createBreak(loc);

        builder.setInsertionPointToEnd(switchBlock);

        // A failed compare-exchange is a read-only operation.  So
        // memory_order_release and memory_order_acq_rel are not supported for
        // the failure memory order.  They fall back to memory_order_relaxed.

        // case seq_cst:
        emitSingleMemOrderCase(builder, loc, FailureOrderVal.getType(),
                               cir::MemOrder::SequentiallyConsistent);
        emitAtomicCmpXchg(CGF, E, IsWeak, Dest, Ptr, Val1, Val2, Size,
                          SuccessOrder, cir::MemOrder::SequentiallyConsistent,
                          Scope);
        builder.createBreak(loc);

        builder.setInsertionPointToEnd(switchBlock);
        builder.createYield(loc);
      });
}

static void emitAtomicOp(CIRGenFunction &CGF, AtomicExpr *E, Address Dest,
                         Address Ptr, Address Val1, Address Val2,
                         mlir::Value IsWeak, mlir::Value FailureOrder,
                         uint64_t Size, cir::MemOrder Order,
                         cir::MemScopeKind Scope) {
  assert(!cir::MissingFeatures::syncScopeID());
  StringRef Op;

  auto &builder = CGF.getBuilder();
  auto loc = CGF.getLoc(E->getSourceRange());
  auto orderAttr = cir::MemOrderAttr::get(builder.getContext(), Order);
  cir::AtomicFetchKindAttr fetchAttr;
  bool fetchFirst = true;

  switch (E->getOp()) {
  case AtomicExpr::AO__c11_atomic_init:
  case AtomicExpr::AO__opencl_atomic_init:
    llvm_unreachable("Already handled!");

  case AtomicExpr::AO__c11_atomic_compare_exchange_strong:
  case AtomicExpr::AO__hip_atomic_compare_exchange_strong:
  case AtomicExpr::AO__opencl_atomic_compare_exchange_strong:
    emitAtomicCmpXchgFailureSet(CGF, E, false, Dest, Ptr, Val1, Val2,
                                FailureOrder, Size, Order, Scope);
    return;
  case AtomicExpr::AO__c11_atomic_compare_exchange_weak:
  case AtomicExpr::AO__opencl_atomic_compare_exchange_weak:
  case AtomicExpr::AO__hip_atomic_compare_exchange_weak:
    emitAtomicCmpXchgFailureSet(CGF, E, true, Dest, Ptr, Val1, Val2,
                                FailureOrder, Size, Order, Scope);
    return;
  case AtomicExpr::AO__atomic_compare_exchange:
  case AtomicExpr::AO__atomic_compare_exchange_n:
  case AtomicExpr::AO__scoped_atomic_compare_exchange:
  case AtomicExpr::AO__scoped_atomic_compare_exchange_n: {
    bool weakVal;
    if (isCstWeak(IsWeak, weakVal)) {
      emitAtomicCmpXchgFailureSet(CGF, E, weakVal, Dest, Ptr, Val1, Val2,
                                  FailureOrder, Size, Order, Scope);
    } else {
      llvm_unreachable("NYI");
    }
    return;
  }
  case AtomicExpr::AO__c11_atomic_load:
  case AtomicExpr::AO__opencl_atomic_load:
  case AtomicExpr::AO__hip_atomic_load:
  case AtomicExpr::AO__atomic_load_n:
  case AtomicExpr::AO__atomic_load:
  case AtomicExpr::AO__scoped_atomic_load_n:
  case AtomicExpr::AO__scoped_atomic_load: {
    auto load = builder.createLoad(loc, Ptr);
    // FIXME(cir): add scope information.
    assert(!cir::MissingFeatures::syncScopeID());
    load->setAttr("mem_order", orderAttr);
    if (E->isVolatile())
      load->setAttr("is_volatile", mlir::UnitAttr::get(builder.getContext()));

    // TODO(cir): this logic should be part of createStore, but doing so
    // currently breaks CodeGen/union.cpp and CodeGen/union.cpp.
    auto ptrTy = mlir::cast<cir::PointerType>(Dest.getPointer().getType());
    if (Dest.getElementType() != ptrTy.getPointee()) {
      Dest = Dest.withPointer(
          builder.createPtrBitcast(Dest.getPointer(), Dest.getElementType()));
    }
    builder.createStore(loc, load->getResult(0), Dest);
    return;
  }

  case AtomicExpr::AO__c11_atomic_store:
  case AtomicExpr::AO__opencl_atomic_store:
  case AtomicExpr::AO__hip_atomic_store:
  case AtomicExpr::AO__atomic_store:
  case AtomicExpr::AO__atomic_store_n:
  case AtomicExpr::AO__scoped_atomic_store:
  case AtomicExpr::AO__scoped_atomic_store_n: {
    auto loadVal1 = builder.createLoad(loc, Val1);
    // FIXME(cir): add scope information.
    assert(!cir::MissingFeatures::syncScopeID());
    builder.createStore(loc, loadVal1, Ptr, E->isVolatile(),
                        /*alignment=*/mlir::IntegerAttr{}, orderAttr);
    return;
  }

  case AtomicExpr::AO__c11_atomic_exchange:
  case AtomicExpr::AO__hip_atomic_exchange:
  case AtomicExpr::AO__opencl_atomic_exchange:
  case AtomicExpr::AO__atomic_exchange_n:
  case AtomicExpr::AO__atomic_exchange:
  case AtomicExpr::AO__scoped_atomic_exchange_n:
  case AtomicExpr::AO__scoped_atomic_exchange:
    Op = cir::AtomicXchg::getOperationName();
    break;

  case AtomicExpr::AO__atomic_add_fetch:
  case AtomicExpr::AO__scoped_atomic_add_fetch:
    fetchFirst = false;
    [[fallthrough]];
  case AtomicExpr::AO__c11_atomic_fetch_add:
  case AtomicExpr::AO__hip_atomic_fetch_add:
  case AtomicExpr::AO__opencl_atomic_fetch_add:
  case AtomicExpr::AO__atomic_fetch_add:
  case AtomicExpr::AO__scoped_atomic_fetch_add:
    Op = cir::AtomicFetch::getOperationName();
    fetchAttr = cir::AtomicFetchKindAttr::get(builder.getContext(),
                                              cir::AtomicFetchKind::Add);
    break;

  case AtomicExpr::AO__atomic_sub_fetch:
  case AtomicExpr::AO__scoped_atomic_sub_fetch:
    fetchFirst = false;
    [[fallthrough]];
  case AtomicExpr::AO__c11_atomic_fetch_sub:
  case AtomicExpr::AO__hip_atomic_fetch_sub:
  case AtomicExpr::AO__opencl_atomic_fetch_sub:
  case AtomicExpr::AO__atomic_fetch_sub:
  case AtomicExpr::AO__scoped_atomic_fetch_sub:
    Op = cir::AtomicFetch::getOperationName();
    fetchAttr = cir::AtomicFetchKindAttr::get(builder.getContext(),
                                              cir::AtomicFetchKind::Sub);
    break;

  case AtomicExpr::AO__atomic_min_fetch:
  case AtomicExpr::AO__scoped_atomic_min_fetch:
    fetchFirst = false;
    [[fallthrough]];
  case AtomicExpr::AO__c11_atomic_fetch_min:
  case AtomicExpr::AO__hip_atomic_fetch_min:
  case AtomicExpr::AO__opencl_atomic_fetch_min:
  case AtomicExpr::AO__atomic_fetch_min:
  case AtomicExpr::AO__scoped_atomic_fetch_min:
    Op = cir::AtomicFetch::getOperationName();
    fetchAttr = cir::AtomicFetchKindAttr::get(builder.getContext(),
                                              cir::AtomicFetchKind::Min);
    break;

  case AtomicExpr::AO__atomic_max_fetch:
  case AtomicExpr::AO__scoped_atomic_max_fetch:
    fetchFirst = false;
    [[fallthrough]];
  case AtomicExpr::AO__c11_atomic_fetch_max:
  case AtomicExpr::AO__hip_atomic_fetch_max:
  case AtomicExpr::AO__opencl_atomic_fetch_max:
  case AtomicExpr::AO__atomic_fetch_max:
  case AtomicExpr::AO__scoped_atomic_fetch_max:
    Op = cir::AtomicFetch::getOperationName();
    fetchAttr = cir::AtomicFetchKindAttr::get(builder.getContext(),
                                              cir::AtomicFetchKind::Max);
    break;

  case AtomicExpr::AO__atomic_and_fetch:
  case AtomicExpr::AO__scoped_atomic_and_fetch:
    fetchFirst = false;
    [[fallthrough]];
  case AtomicExpr::AO__c11_atomic_fetch_and:
  case AtomicExpr::AO__hip_atomic_fetch_and:
  case AtomicExpr::AO__opencl_atomic_fetch_and:
  case AtomicExpr::AO__atomic_fetch_and:
  case AtomicExpr::AO__scoped_atomic_fetch_and:
    Op = cir::AtomicFetch::getOperationName();
    fetchAttr = cir::AtomicFetchKindAttr::get(builder.getContext(),
                                              cir::AtomicFetchKind::And);
    break;

  case AtomicExpr::AO__atomic_or_fetch:
  case AtomicExpr::AO__scoped_atomic_or_fetch:
    fetchFirst = false;
    [[fallthrough]];
  case AtomicExpr::AO__c11_atomic_fetch_or:
  case AtomicExpr::AO__hip_atomic_fetch_or:
  case AtomicExpr::AO__opencl_atomic_fetch_or:
  case AtomicExpr::AO__atomic_fetch_or:
  case AtomicExpr::AO__scoped_atomic_fetch_or:
    Op = cir::AtomicFetch::getOperationName();
    fetchAttr = cir::AtomicFetchKindAttr::get(builder.getContext(),
                                              cir::AtomicFetchKind::Or);
    break;

  case AtomicExpr::AO__atomic_xor_fetch:
  case AtomicExpr::AO__scoped_atomic_xor_fetch:
    fetchFirst = false;
    [[fallthrough]];
  case AtomicExpr::AO__c11_atomic_fetch_xor:
  case AtomicExpr::AO__hip_atomic_fetch_xor:
  case AtomicExpr::AO__opencl_atomic_fetch_xor:
  case AtomicExpr::AO__atomic_fetch_xor:
  case AtomicExpr::AO__scoped_atomic_fetch_xor:
    Op = cir::AtomicFetch::getOperationName();
    fetchAttr = cir::AtomicFetchKindAttr::get(builder.getContext(),
                                              cir::AtomicFetchKind::Xor);
    break;

  case AtomicExpr::AO__atomic_nand_fetch:
  case AtomicExpr::AO__scoped_atomic_nand_fetch:
    fetchFirst = false;
    [[fallthrough]];
  case AtomicExpr::AO__c11_atomic_fetch_nand:
  case AtomicExpr::AO__atomic_fetch_nand:
  case AtomicExpr::AO__scoped_atomic_fetch_nand:
    Op = cir::AtomicFetch::getOperationName();
    fetchAttr = cir::AtomicFetchKindAttr::get(builder.getContext(),
                                              cir::AtomicFetchKind::Nand);
    break;
  case AtomicExpr::AO__atomic_test_and_set: {
    llvm_unreachable("NYI");
  }

  case AtomicExpr::AO__atomic_clear: {
    llvm_unreachable("NYI");
  }
  }

  assert(Op.size() && "expected operation name to build");
  auto LoadVal1 = builder.createLoad(loc, Val1);

  SmallVector<mlir::Value> atomicOperands = {Ptr.getPointer(), LoadVal1};
  SmallVector<mlir::Type> atomicResTys = {LoadVal1.getType()};
  auto RMWI = builder.create(loc, builder.getStringAttr(Op), atomicOperands,
                             atomicResTys, {});

  if (fetchAttr)
    RMWI->setAttr("binop", fetchAttr);
  RMWI->setAttr("mem_order", orderAttr);
  if (E->isVolatile())
    RMWI->setAttr("is_volatile", mlir::UnitAttr::get(builder.getContext()));
  if (fetchFirst && Op == cir::AtomicFetch::getOperationName())
    RMWI->setAttr("fetch_first", mlir::UnitAttr::get(builder.getContext()));

  auto Result = RMWI->getResult(0);

  // TODO(cir): this logic should be part of createStore, but doing so currently
  // breaks CodeGen/union.cpp and CodeGen/union.cpp.
  auto ptrTy = mlir::cast<cir::PointerType>(Dest.getPointer().getType());
  if (Dest.getElementType() != ptrTy.getPointee()) {
    Dest = Dest.withPointer(
        builder.createPtrBitcast(Dest.getPointer(), Dest.getElementType()));
  }
  builder.createStore(loc, Result, Dest);
}

static RValue emitAtomicLibcall(CIRGenFunction &CGF, StringRef fnName,
                                QualType resultType, CallArgList &args) {
  [[maybe_unused]] const CIRGenFunctionInfo &fnInfo =
      CGF.CGM.getTypes().arrangeBuiltinFunctionCall(resultType, args);
  [[maybe_unused]] auto fnTy = CGF.CGM.getTypes().GetFunctionType(fnInfo);
  llvm_unreachable("NYI");
}

static void emitAtomicOp(CIRGenFunction &CGF, AtomicExpr *Expr, Address Dest,
                         Address Ptr, Address Val1, Address Val2,
                         mlir::Value IsWeak, mlir::Value FailureOrder,
                         uint64_t Size, cir::MemOrder Order,
                         mlir::Value Scope) {
  auto ScopeModel = Expr->getScopeModel();

  // LLVM atomic instructions always have synch scope. If clang atomic
  // expression has no scope operand, use default LLVM synch scope.
  if (!ScopeModel) {
    assert(!cir::MissingFeatures::syncScopeID());
    emitAtomicOp(CGF, Expr, Dest, Ptr, Val1, Val2, IsWeak, FailureOrder, Size,
                 Order, cir::MemScopeKind::MemScope_System);
    return;
  }

  // Handle constant scope.
  if (getConstOpIntAttr(Scope)) {
    assert(!cir::MissingFeatures::syncScopeID());
    llvm_unreachable("NYI");
    return;
  }

  // Handle non-constant scope.
  llvm_unreachable("NYI");
}

RValue CIRGenFunction::emitAtomicExpr(AtomicExpr *E) {
  QualType AtomicTy = E->getPtr()->getType()->getPointeeType();
  QualType MemTy = AtomicTy;
  if (const AtomicType *AT = AtomicTy->getAs<AtomicType>())
    MemTy = AT->getValueType();
  mlir::Value IsWeak = nullptr, OrderFail = nullptr;

  Address Val1 = Address::invalid();
  Address Val2 = Address::invalid();
  Address Dest = Address::invalid();
  Address Ptr = emitPointerWithAlignment(E->getPtr());

  if (E->getOp() == AtomicExpr::AO__c11_atomic_init ||
      E->getOp() == AtomicExpr::AO__opencl_atomic_init) {
    LValue lvalue = makeAddrLValue(Ptr, AtomicTy);
    emitAtomicInit(E->getVal1(), lvalue);
    return RValue::get(nullptr);
  }

  auto TInfo = getContext().getTypeInfoInChars(AtomicTy);
  uint64_t Size = TInfo.Width.getQuantity();
  unsigned MaxInlineWidthInBits = getTarget().getMaxAtomicInlineWidth();

  CharUnits MaxInlineWidth =
      getContext().toCharUnitsFromBits(MaxInlineWidthInBits);
  DiagnosticsEngine &Diags = CGM.getDiags();
  bool Misaligned = (Ptr.getAlignment() % TInfo.Width) != 0;
  bool Oversized = getContext().toBits(TInfo.Width) > MaxInlineWidthInBits;
  if (Misaligned) {
    Diags.Report(E->getBeginLoc(), diag::warn_atomic_op_misaligned)
        << (int)TInfo.Width.getQuantity()
        << (int)Ptr.getAlignment().getQuantity();
  }
  if (Oversized) {
    Diags.Report(E->getBeginLoc(), diag::warn_atomic_op_oversized)
        << (int)TInfo.Width.getQuantity() << (int)MaxInlineWidth.getQuantity();
  }

  auto Order = emitScalarExpr(E->getOrder());
  auto Scope = E->getScopeModel() ? emitScalarExpr(E->getScope()) : nullptr;
  bool ShouldCastToIntPtrTy = true;

  switch (E->getOp()) {
  case AtomicExpr::AO__c11_atomic_init:
  case AtomicExpr::AO__opencl_atomic_init:
    llvm_unreachable("Already handled above with EmitAtomicInit!");

  case AtomicExpr::AO__atomic_load_n:
  case AtomicExpr::AO__scoped_atomic_load_n:
  case AtomicExpr::AO__c11_atomic_load:
  case AtomicExpr::AO__opencl_atomic_load:
  case AtomicExpr::AO__hip_atomic_load:
  case AtomicExpr::AO__atomic_test_and_set:
  case AtomicExpr::AO__atomic_clear:
    break;

  case AtomicExpr::AO__atomic_load:
  case AtomicExpr::AO__scoped_atomic_load:
    Dest = emitPointerWithAlignment(E->getVal1());
    break;

  case AtomicExpr::AO__atomic_store:
  case AtomicExpr::AO__scoped_atomic_store:
    Val1 = emitPointerWithAlignment(E->getVal1());
    break;

  case AtomicExpr::AO__atomic_exchange:
  case AtomicExpr::AO__scoped_atomic_exchange:
    Val1 = emitPointerWithAlignment(E->getVal1());
    Dest = emitPointerWithAlignment(E->getVal2());
    break;

  case AtomicExpr::AO__atomic_compare_exchange:
  case AtomicExpr::AO__atomic_compare_exchange_n:
  case AtomicExpr::AO__c11_atomic_compare_exchange_weak:
  case AtomicExpr::AO__c11_atomic_compare_exchange_strong:
  case AtomicExpr::AO__hip_atomic_compare_exchange_weak:
  case AtomicExpr::AO__hip_atomic_compare_exchange_strong:
  case AtomicExpr::AO__opencl_atomic_compare_exchange_weak:
  case AtomicExpr::AO__opencl_atomic_compare_exchange_strong:
  case AtomicExpr::AO__scoped_atomic_compare_exchange:
  case AtomicExpr::AO__scoped_atomic_compare_exchange_n:
    Val1 = emitPointerWithAlignment(E->getVal1());
    if (E->getOp() == AtomicExpr::AO__atomic_compare_exchange ||
        E->getOp() == AtomicExpr::AO__scoped_atomic_compare_exchange)
      Val2 = emitPointerWithAlignment(E->getVal2());
    else
      Val2 = emitValToTemp(*this, E->getVal2());
    OrderFail = emitScalarExpr(E->getOrderFail());
    if (E->getOp() == AtomicExpr::AO__atomic_compare_exchange_n ||
        E->getOp() == AtomicExpr::AO__atomic_compare_exchange ||
        E->getOp() == AtomicExpr::AO__scoped_atomic_compare_exchange_n ||
        E->getOp() == AtomicExpr::AO__scoped_atomic_compare_exchange) {
      IsWeak = emitScalarExpr(E->getWeak());
    }
    break;

  case AtomicExpr::AO__c11_atomic_fetch_add:
  case AtomicExpr::AO__c11_atomic_fetch_sub:
  case AtomicExpr::AO__hip_atomic_fetch_add:
  case AtomicExpr::AO__hip_atomic_fetch_sub:
  case AtomicExpr::AO__opencl_atomic_fetch_add:
  case AtomicExpr::AO__opencl_atomic_fetch_sub:
    if (MemTy->isPointerType()) {
      llvm_unreachable("NYI");
    }
    [[fallthrough]];
  case AtomicExpr::AO__atomic_fetch_add:
  case AtomicExpr::AO__atomic_fetch_max:
  case AtomicExpr::AO__atomic_fetch_min:
  case AtomicExpr::AO__atomic_fetch_sub:
  case AtomicExpr::AO__atomic_add_fetch:
  case AtomicExpr::AO__atomic_max_fetch:
  case AtomicExpr::AO__atomic_min_fetch:
  case AtomicExpr::AO__atomic_sub_fetch:
  case AtomicExpr::AO__c11_atomic_fetch_max:
  case AtomicExpr::AO__c11_atomic_fetch_min:
  case AtomicExpr::AO__opencl_atomic_fetch_max:
  case AtomicExpr::AO__opencl_atomic_fetch_min:
  case AtomicExpr::AO__hip_atomic_fetch_max:
  case AtomicExpr::AO__hip_atomic_fetch_min:
  case AtomicExpr::AO__scoped_atomic_fetch_add:
  case AtomicExpr::AO__scoped_atomic_fetch_max:
  case AtomicExpr::AO__scoped_atomic_fetch_min:
  case AtomicExpr::AO__scoped_atomic_fetch_sub:
  case AtomicExpr::AO__scoped_atomic_add_fetch:
  case AtomicExpr::AO__scoped_atomic_max_fetch:
  case AtomicExpr::AO__scoped_atomic_min_fetch:
  case AtomicExpr::AO__scoped_atomic_sub_fetch:
    ShouldCastToIntPtrTy = !MemTy->isFloatingType();
    [[fallthrough]];

  case AtomicExpr::AO__atomic_fetch_and:
  case AtomicExpr::AO__atomic_fetch_nand:
  case AtomicExpr::AO__atomic_fetch_or:
  case AtomicExpr::AO__atomic_fetch_xor:
  case AtomicExpr::AO__atomic_and_fetch:
  case AtomicExpr::AO__atomic_nand_fetch:
  case AtomicExpr::AO__atomic_or_fetch:
  case AtomicExpr::AO__atomic_xor_fetch:
  case AtomicExpr::AO__atomic_store_n:
  case AtomicExpr::AO__atomic_exchange_n:
  case AtomicExpr::AO__c11_atomic_fetch_and:
  case AtomicExpr::AO__c11_atomic_fetch_nand:
  case AtomicExpr::AO__c11_atomic_fetch_or:
  case AtomicExpr::AO__c11_atomic_fetch_xor:
  case AtomicExpr::AO__c11_atomic_store:
  case AtomicExpr::AO__c11_atomic_exchange:
  case AtomicExpr::AO__hip_atomic_fetch_and:
  case AtomicExpr::AO__hip_atomic_fetch_or:
  case AtomicExpr::AO__hip_atomic_fetch_xor:
  case AtomicExpr::AO__hip_atomic_store:
  case AtomicExpr::AO__hip_atomic_exchange:
  case AtomicExpr::AO__opencl_atomic_fetch_and:
  case AtomicExpr::AO__opencl_atomic_fetch_or:
  case AtomicExpr::AO__opencl_atomic_fetch_xor:
  case AtomicExpr::AO__opencl_atomic_store:
  case AtomicExpr::AO__opencl_atomic_exchange:
  case AtomicExpr::AO__scoped_atomic_fetch_and:
  case AtomicExpr::AO__scoped_atomic_fetch_nand:
  case AtomicExpr::AO__scoped_atomic_fetch_or:
  case AtomicExpr::AO__scoped_atomic_fetch_xor:
  case AtomicExpr::AO__scoped_atomic_and_fetch:
  case AtomicExpr::AO__scoped_atomic_nand_fetch:
  case AtomicExpr::AO__scoped_atomic_or_fetch:
  case AtomicExpr::AO__scoped_atomic_xor_fetch:
  case AtomicExpr::AO__scoped_atomic_store_n:
  case AtomicExpr::AO__scoped_atomic_exchange_n:
    Val1 = emitValToTemp(*this, E->getVal1());
    break;
  }

  QualType RValTy = E->getType().getUnqualifiedType();

  // The inlined atomics only function on iN types, where N is a power of 2. We
  // need to make sure (via temporaries if necessary) that all incoming values
  // are compatible.
  LValue AtomicVal = makeAddrLValue(Ptr, AtomicTy);
  AtomicInfo Atomics(*this, AtomicVal, getLoc(E->getSourceRange()));

  if (ShouldCastToIntPtrTy) {
    Ptr = Atomics.castToAtomicIntPointer(Ptr);
    if (Val1.isValid())
      Val1 = Atomics.convertToAtomicIntPointer(Val1);
    if (Val2.isValid())
      Val2 = Atomics.convertToAtomicIntPointer(Val2);
  }
  if (Dest.isValid()) {
    if (ShouldCastToIntPtrTy)
      Dest = Atomics.castToAtomicIntPointer(Dest);
  } else if (E->isCmpXChg())
    Dest = CreateMemTemp(RValTy, getLoc(E->getSourceRange()), "cmpxchg.bool");
  else if (!RValTy->isVoidType()) {
    Dest = Atomics.CreateTempAlloca();
    if (ShouldCastToIntPtrTy)
      Dest = Atomics.castToAtomicIntPointer(Dest);
  }

  bool PowerOf2Size = (Size & (Size - 1)) == 0;
  bool UseLibcall = !PowerOf2Size || (Size > 16);

  // For atomics larger than 16 bytes, emit a libcall from the frontend. This
  // avoids the overhead of dealing with excessively-large value types in IR.
  // Non-power-of-2 values also lower to libcall here, as they are not currently
  // permitted in IR instructions (although that constraint could be relaxed in
  // the future). For other cases where a libcall is required on a given
  // platform, we let the backend handle it (this includes handling for all of
  // the size-optimized libcall variants, which are only valid up to 16 bytes.)
  //
  // See: https://llvm.org/docs/Atomics.html#libcalls-atomic
  if (UseLibcall) {
    CallArgList Args;
    // For non-optimized library calls, the size is the first parameter.
    Args.add(RValue::get(builder.getConstInt(getLoc(E->getSourceRange()),
                                             SizeTy, Size)),
             getContext().getSizeType());

    // The atomic address is the second parameter.
    // The OpenCL atomic library functions only accept pointer arguments to
    // generic address space.
    auto CastToGenericAddrSpace = [&](mlir::Value V, QualType PT) {
      if (!E->isOpenCL())
        return V;
      llvm_unreachable("NYI");
    };

    Args.add(RValue::get(CastToGenericAddrSpace(Ptr.emitRawPointer(),
                                                E->getPtr()->getType())),
             getContext().VoidPtrTy);

    // The next 1-3 parameters are op-dependent.
    std::string LibCallName;
    QualType RetTy;
    bool HaveRetTy = false;
    switch (E->getOp()) {
    case AtomicExpr::AO__c11_atomic_init:
    case AtomicExpr::AO__opencl_atomic_init:
      llvm_unreachable("Already handled!");

    // There is only one libcall for compare an exchange, because there is no
    // optimisation benefit possible from a libcall version of a weak compare
    // and exchange.
    // bool __atomic_compare_exchange(size_t size, void *mem, void *expected,
    //                                void *desired, int success, int failure)
    case AtomicExpr::AO__atomic_compare_exchange:
    case AtomicExpr::AO__atomic_compare_exchange_n:
    case AtomicExpr::AO__c11_atomic_compare_exchange_weak:
    case AtomicExpr::AO__c11_atomic_compare_exchange_strong:
    case AtomicExpr::AO__hip_atomic_compare_exchange_weak:
    case AtomicExpr::AO__hip_atomic_compare_exchange_strong:
    case AtomicExpr::AO__opencl_atomic_compare_exchange_weak:
    case AtomicExpr::AO__opencl_atomic_compare_exchange_strong:
    case AtomicExpr::AO__scoped_atomic_compare_exchange:
    case AtomicExpr::AO__scoped_atomic_compare_exchange_n:
      LibCallName = "__atomic_compare_exchange";
      llvm_unreachable("NYI");
      break;
    // void __atomic_exchange(size_t size, void *mem, void *val, void *return,
    //                        int order)
    case AtomicExpr::AO__atomic_exchange:
    case AtomicExpr::AO__atomic_exchange_n:
    case AtomicExpr::AO__c11_atomic_exchange:
    case AtomicExpr::AO__hip_atomic_exchange:
    case AtomicExpr::AO__opencl_atomic_exchange:
    case AtomicExpr::AO__scoped_atomic_exchange:
    case AtomicExpr::AO__scoped_atomic_exchange_n:
      LibCallName = "__atomic_exchange";
      llvm_unreachable("NYI");
      break;
    // void __atomic_store(size_t size, void *mem, void *val, int order)
    case AtomicExpr::AO__atomic_store:
    case AtomicExpr::AO__atomic_store_n:
    case AtomicExpr::AO__c11_atomic_store:
    case AtomicExpr::AO__hip_atomic_store:
    case AtomicExpr::AO__opencl_atomic_store:
    case AtomicExpr::AO__scoped_atomic_store:
    case AtomicExpr::AO__scoped_atomic_store_n:
      LibCallName = "__atomic_store";
      llvm_unreachable("NYI");
      break;
    // void __atomic_load(size_t size, void *mem, void *return, int order)
    case AtomicExpr::AO__atomic_load:
    case AtomicExpr::AO__atomic_load_n:
    case AtomicExpr::AO__c11_atomic_load:
    case AtomicExpr::AO__hip_atomic_load:
    case AtomicExpr::AO__opencl_atomic_load:
    case AtomicExpr::AO__scoped_atomic_load:
    case AtomicExpr::AO__scoped_atomic_load_n:
      LibCallName = "__atomic_load";
      break;
    case AtomicExpr::AO__atomic_add_fetch:
    case AtomicExpr::AO__scoped_atomic_add_fetch:
    case AtomicExpr::AO__atomic_fetch_add:
    case AtomicExpr::AO__c11_atomic_fetch_add:
    case AtomicExpr::AO__hip_atomic_fetch_add:
    case AtomicExpr::AO__opencl_atomic_fetch_add:
    case AtomicExpr::AO__scoped_atomic_fetch_add:
    case AtomicExpr::AO__atomic_and_fetch:
    case AtomicExpr::AO__scoped_atomic_and_fetch:
    case AtomicExpr::AO__atomic_fetch_and:
    case AtomicExpr::AO__c11_atomic_fetch_and:
    case AtomicExpr::AO__hip_atomic_fetch_and:
    case AtomicExpr::AO__opencl_atomic_fetch_and:
    case AtomicExpr::AO__scoped_atomic_fetch_and:
    case AtomicExpr::AO__atomic_or_fetch:
    case AtomicExpr::AO__scoped_atomic_or_fetch:
    case AtomicExpr::AO__atomic_fetch_or:
    case AtomicExpr::AO__c11_atomic_fetch_or:
    case AtomicExpr::AO__hip_atomic_fetch_or:
    case AtomicExpr::AO__opencl_atomic_fetch_or:
    case AtomicExpr::AO__scoped_atomic_fetch_or:
    case AtomicExpr::AO__atomic_sub_fetch:
    case AtomicExpr::AO__scoped_atomic_sub_fetch:
    case AtomicExpr::AO__atomic_fetch_sub:
    case AtomicExpr::AO__c11_atomic_fetch_sub:
    case AtomicExpr::AO__hip_atomic_fetch_sub:
    case AtomicExpr::AO__opencl_atomic_fetch_sub:
    case AtomicExpr::AO__scoped_atomic_fetch_sub:
    case AtomicExpr::AO__atomic_xor_fetch:
    case AtomicExpr::AO__scoped_atomic_xor_fetch:
    case AtomicExpr::AO__atomic_fetch_xor:
    case AtomicExpr::AO__c11_atomic_fetch_xor:
    case AtomicExpr::AO__hip_atomic_fetch_xor:
    case AtomicExpr::AO__opencl_atomic_fetch_xor:
    case AtomicExpr::AO__scoped_atomic_fetch_xor:
    case AtomicExpr::AO__atomic_nand_fetch:
    case AtomicExpr::AO__atomic_fetch_nand:
    case AtomicExpr::AO__c11_atomic_fetch_nand:
    case AtomicExpr::AO__scoped_atomic_fetch_nand:
    case AtomicExpr::AO__scoped_atomic_nand_fetch:
    case AtomicExpr::AO__atomic_min_fetch:
    case AtomicExpr::AO__atomic_fetch_min:
    case AtomicExpr::AO__c11_atomic_fetch_min:
    case AtomicExpr::AO__hip_atomic_fetch_min:
    case AtomicExpr::AO__opencl_atomic_fetch_min:
    case AtomicExpr::AO__scoped_atomic_fetch_min:
    case AtomicExpr::AO__scoped_atomic_min_fetch:
    case AtomicExpr::AO__atomic_max_fetch:
    case AtomicExpr::AO__atomic_fetch_max:
    case AtomicExpr::AO__c11_atomic_fetch_max:
    case AtomicExpr::AO__hip_atomic_fetch_max:
    case AtomicExpr::AO__opencl_atomic_fetch_max:
    case AtomicExpr::AO__scoped_atomic_fetch_max:
    case AtomicExpr::AO__scoped_atomic_max_fetch:
    case AtomicExpr::AO__atomic_test_and_set:
    case AtomicExpr::AO__atomic_clear:
      llvm_unreachable("Integral atomic operations always become atomicrmw!");
    }

    if (E->isOpenCL()) {
      LibCallName =
          std::string("__opencl") + StringRef(LibCallName).drop_front(1).str();
    }
    // By default, assume we return a value of the atomic type.
    if (!HaveRetTy) {
      llvm_unreachable("NYI");
    }
    // Order is always the last parameter.
    Args.add(RValue::get(Order), getContext().IntTy);
    if (E->isOpenCL()) {
      llvm_unreachable("NYI");
    }

    [[maybe_unused]] RValue Res =
        emitAtomicLibcall(*this, LibCallName, RetTy, Args);
    // The value is returned directly from the libcall.
    if (E->isCmpXChg()) {
      llvm_unreachable("NYI");
    }

    if (RValTy->isVoidType()) {
      llvm_unreachable("NYI");
    }

    llvm_unreachable("NYI");
  }

  bool IsStore = E->getOp() == AtomicExpr::AO__c11_atomic_store ||
                 E->getOp() == AtomicExpr::AO__opencl_atomic_store ||
                 E->getOp() == AtomicExpr::AO__hip_atomic_store ||
                 E->getOp() == AtomicExpr::AO__atomic_store ||
                 E->getOp() == AtomicExpr::AO__atomic_store_n ||
                 E->getOp() == AtomicExpr::AO__scoped_atomic_store ||
                 E->getOp() == AtomicExpr::AO__scoped_atomic_store_n ||
                 E->getOp() == AtomicExpr::AO__atomic_clear;
  bool IsLoad = E->getOp() == AtomicExpr::AO__c11_atomic_load ||
                E->getOp() == AtomicExpr::AO__opencl_atomic_load ||
                E->getOp() == AtomicExpr::AO__hip_atomic_load ||
                E->getOp() == AtomicExpr::AO__atomic_load ||
                E->getOp() == AtomicExpr::AO__atomic_load_n ||
                E->getOp() == AtomicExpr::AO__scoped_atomic_load ||
                E->getOp() == AtomicExpr::AO__scoped_atomic_load_n;

  if (auto ordAttr = getConstOpIntAttr(Order)) {
    // We should not ever get to a case where the ordering isn't a valid CABI
    // value, but it's hard to enforce that in general.
    auto ord = ordAttr.getUInt();
    if (cir::isValidCIRAtomicOrderingCABI(ord)) {
      switch ((cir::MemOrder)ord) {
      case cir::MemOrder::Relaxed:
        emitAtomicOp(*this, E, Dest, Ptr, Val1, Val2, IsWeak, OrderFail, Size,
                     cir::MemOrder::Relaxed, Scope);
        break;
      case cir::MemOrder::Consume:
      case cir::MemOrder::Acquire:
        if (IsStore)
          break; // Avoid crashing on code with undefined behavior
        emitAtomicOp(*this, E, Dest, Ptr, Val1, Val2, IsWeak, OrderFail, Size,
                     cir::MemOrder::Acquire, Scope);
        break;
      case cir::MemOrder::Release:
        if (IsLoad)
          break; // Avoid crashing on code with undefined behavior
        emitAtomicOp(*this, E, Dest, Ptr, Val1, Val2, IsWeak, OrderFail, Size,
                     cir::MemOrder::Release, Scope);
        break;
      case cir::MemOrder::AcquireRelease:
        if (IsLoad || IsStore)
          break; // Avoid crashing on code with undefined behavior
        emitAtomicOp(*this, E, Dest, Ptr, Val1, Val2, IsWeak, OrderFail, Size,
                     cir::MemOrder::AcquireRelease, Scope);
        break;
      case cir::MemOrder::SequentiallyConsistent:
        emitAtomicOp(*this, E, Dest, Ptr, Val1, Val2, IsWeak, OrderFail, Size,
                     cir::MemOrder::SequentiallyConsistent, Scope);
        break;
      }
    }
    if (RValTy->isVoidType())
      return RValue::get(nullptr);

    return convertTempToRValue(
        Dest.withElementType(builder, convertTypeForMem(RValTy)), RValTy,
        E->getExprLoc());
  }

  // The memory order is not known at compile-time.  The atomic operations
  // can't handle runtime memory orders; the memory order must be hard coded.
  // Generate a "switch" statement that converts a runtime value into a
  // compile-time value.
  builder.create<cir::SwitchOp>(
      Order.getLoc(), Order,
      [&](mlir::OpBuilder &b, mlir::Location loc, mlir::OperationState &os) {
        mlir::Block *switchBlock = builder.getBlock();

        // default:
        // Use memory_order_relaxed for relaxed operations and for any memory
        // order value that is not supported.  There is no good way to report
        // an unsupported memory order at runtime, hence the fallback to
        // memory_order_relaxed.
        emitDefaultCase(builder, loc);
        emitAtomicOp(*this, E, Dest, Ptr, Val1, Val2, IsWeak, OrderFail, Size,
                     cir::MemOrder::Relaxed, Scope);
        builder.createBreak(loc);

        builder.setInsertionPointToEnd(switchBlock);

        if (!IsStore) {
          // case consume:
          // case acquire:
          // memory_order_consume is not implemented; it is always treated like
          // memory_order_acquire.  These memory orders are not valid for
          // write-only operations.
          emitDoubleMemOrderCase(builder, loc, Order.getType(),
                                 cir::MemOrder::Consume,
                                 cir::MemOrder::Acquire);
          emitAtomicOp(*this, E, Dest, Ptr, Val1, Val2, IsWeak, OrderFail, Size,
                       cir::MemOrder::Acquire, Scope);
          builder.createBreak(loc);
        }

        builder.setInsertionPointToEnd(switchBlock);

        if (!IsLoad) {
          // case release:
          // memory_order_release is not valid for read-only operations.
          emitSingleMemOrderCase(builder, loc, Order.getType(),
                                 cir::MemOrder::Release);
          emitAtomicOp(*this, E, Dest, Ptr, Val1, Val2, IsWeak, OrderFail, Size,
                       cir::MemOrder::Release, Scope);
          builder.createBreak(loc);
        }

        builder.setInsertionPointToEnd(switchBlock);

        if (!IsLoad && !IsStore) {
          // case acq_rel:
          // memory_order_acq_rel is only valid for read-write operations.
          emitSingleMemOrderCase(builder, loc, Order.getType(),
                                 cir::MemOrder::AcquireRelease);
          emitAtomicOp(*this, E, Dest, Ptr, Val1, Val2, IsWeak, OrderFail, Size,
                       cir::MemOrder::AcquireRelease, Scope);
          builder.createBreak(loc);
        }

        builder.setInsertionPointToEnd(switchBlock);

        // case seq_cst:
        emitSingleMemOrderCase(builder, loc, Order.getType(),
                               cir::MemOrder::SequentiallyConsistent);
        emitAtomicOp(*this, E, Dest, Ptr, Val1, Val2, IsWeak, OrderFail, Size,
                     cir::MemOrder::SequentiallyConsistent, Scope);
        builder.createBreak(loc);

        builder.setInsertionPointToEnd(switchBlock);
        builder.createYield(loc);
      });

  if (RValTy->isVoidType())
    return RValue::get(nullptr);

  return convertTempToRValue(
      Dest.withElementType(builder, convertTypeForMem(RValTy)), RValTy,
      E->getExprLoc());
}

void CIRGenFunction::emitAtomicStore(RValue rvalue, LValue lvalue,
                                     bool isInit) {
  bool IsVolatile = lvalue.isVolatileQualified();
  cir::MemOrder MO;
  if (lvalue.getType()->isAtomicType()) {
    MO = cir::MemOrder::SequentiallyConsistent;
  } else {
    MO = cir::MemOrder::Release;
    IsVolatile = true;
  }
  return emitAtomicStore(rvalue, lvalue, MO, IsVolatile, isInit);
}

/// Return true if \param ValTy is a type that should be casted to integer
/// around the atomic memory operation. If \param CmpXchg is true, then the
/// cast of a floating point type is made as that instruction can not have
/// floating point operands.  TODO: Allow compare-and-exchange and FP - see
/// comment in CIRGenAtomicExpandPass.cpp.
static bool shouldCastToInt(mlir::Type ValTy, bool CmpXchg) {
  if (cir::isAnyFloatingPointType(ValTy))
    return isa<cir::FP80Type>(ValTy) || CmpXchg;
  return !isa<cir::IntType>(ValTy) && !isa<cir::PointerType>(ValTy);
}

mlir::Value AtomicInfo::getScalarRValValueOrNull(RValue RVal) const {
  if (RVal.isScalar() && (!hasPadding() || !LVal.isSimple()))
    return RVal.getScalarVal();
  return nullptr;
}

/// Materialize an r-value into memory for the purposes of storing it
/// to an atomic type.
Address AtomicInfo::materializeRValue(RValue rvalue) const {
  // Aggregate r-values are already in memory, and EmitAtomicStore
  // requires them to be values of the atomic type.
  if (rvalue.isAggregate())
    return rvalue.getAggregateAddress();

  // Otherwise, make a temporary and materialize into it.
  LValue TempLV = CGF.makeAddrLValue(CreateTempAlloca(), getAtomicType());
  AtomicInfo Atomics(CGF, TempLV, TempLV.getAddress().getPointer().getLoc());
  Atomics.emitCopyIntoMemory(rvalue);
  return TempLV.getAddress();
}

bool AtomicInfo::emitMemSetZeroIfNecessary() const {
  assert(LVal.isSimple());
  Address addr = LVal.getAddress();
  if (!requiresMemSetZero(addr.getElementType()))
    return false;

  llvm_unreachable("NYI");
}

/// Copy an r-value into memory as part of storing to an atomic type.
/// This needs to create a bit-pattern suitable for atomic operations.
void AtomicInfo::emitCopyIntoMemory(RValue rvalue) const {
  assert(LVal.isSimple());
  // If we have an r-value, the rvalue should be of the atomic type,
  // which means that the caller is responsible for having zeroed
  // any padding.  Just do an aggregate copy of that type.
  if (rvalue.isAggregate()) {
    llvm_unreachable("NYI");
    return;
  }

  // Okay, otherwise we're copying stuff.

  // Zero out the buffer if necessary.
  emitMemSetZeroIfNecessary();

  // Drill past the padding if present.
  LValue TempLVal = projectValue();

  // Okay, store the rvalue in.
  if (rvalue.isScalar()) {
    CGF.emitStoreOfScalar(rvalue.getScalarVal(), TempLVal, /*init*/ true);
  } else {
    llvm_unreachable("NYI");
  }
}

mlir::Value AtomicInfo::convertRValueToInt(RValue RVal, bool CmpXchg) const {
  // If we've got a scalar value of the right size, try to avoid going
  // through memory. Floats get casted if needed by AtomicExpandPass.
  if (auto Value = getScalarRValValueOrNull(RVal)) {
    if (!shouldCastToInt(Value.getType(), CmpXchg)) {
      return CGF.emitToMemory(Value, ValueTy);
    } else {
      llvm_unreachable("NYI");
    }
  }

  llvm_unreachable("NYI");
}

/// Emit a store to an l-value of atomic type.
///
/// Note that the r-value is expected to be an r-value *of the atomic
/// type*; this means that for aggregate r-values, it should include
/// storage for any padding that was necessary.
void CIRGenFunction::emitAtomicStore(RValue rvalue, LValue dest,
                                     cir::MemOrder MO, bool IsVolatile,
                                     bool isInit) {
  // If this is an aggregate r-value, it should agree in type except
  // maybe for address-space qualification.
  auto loc = dest.getPointer().getLoc();
  assert(!rvalue.isAggregate() ||
         rvalue.getAggregateAddress().getElementType() ==
             dest.getAddress().getElementType());

  AtomicInfo atomics(*this, dest, loc);
  LValue LVal = atomics.getAtomicLValue();

  // If this is an initialization, just put the value there normally.
  if (LVal.isSimple()) {
    if (isInit) {
      atomics.emitCopyIntoMemory(rvalue);
      return;
    }

    // Check whether we should use a library call.
    if (atomics.shouldUseLibcall()) {
      llvm_unreachable("NYI");
    }

    // Okay, we're doing this natively.
    auto ValToStore = atomics.convertRValueToInt(rvalue);

    // Do the atomic store.
    Address Addr = atomics.getAtomicAddress();
    if (auto Value = atomics.getScalarRValValueOrNull(rvalue))
      if (shouldCastToInt(Value.getType(), /*CmpXchg=*/false)) {
        Addr = atomics.castToAtomicIntPointer(Addr);
        ValToStore = builder.createIntCast(ValToStore, Addr.getElementType());
      }
    auto store = builder.createStore(loc, ValToStore, Addr);

    if (MO == cir::MemOrder::Acquire)
      MO = cir::MemOrder::Relaxed; // Monotonic
    else if (MO == cir::MemOrder::AcquireRelease)
      MO = cir::MemOrder::Release;
    // Initializations don't need to be atomic.
    if (!isInit)
      store.setAtomic(MO);

    // Other decoration.
    if (IsVolatile)
      store.setIsVolatile(true);

    CGM.decorateOperationWithTBAA(store, dest.getTBAAInfo());
    return;
  }

  llvm_unreachable("NYI");
}

void CIRGenFunction::emitAtomicInit(Expr *init, LValue dest) {
  AtomicInfo atomics(*this, dest, getLoc(init->getSourceRange()));

  switch (atomics.getEvaluationKind()) {
  case cir::TEK_Scalar: {
    mlir::Value value = emitScalarExpr(init);
    atomics.emitCopyIntoMemory(RValue::get(value));
    return;
  }

  case cir::TEK_Complex: {
    llvm_unreachable("NYI");
    return;
  }

  case cir::TEK_Aggregate: {
    // Fix up the destination if the initializer isn't an expression
    // of atomic type.
    llvm_unreachable("NYI");
    return;
  }
  }
  llvm_unreachable("bad evaluation kind");
}
