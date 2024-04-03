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
#include "CIRDataLayout.h"
#include "CIRGenFunction.h"
#include "CIRGenModule.h"
#include "CIRGenOpenMPRuntime.h"
#include "TargetInfo.h"
#include "UnimplementedFeatureGuarding.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CodeGen/CGFunctionInfo.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstdint>

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"

using namespace cir;
using namespace clang;

namespace {
class AtomicInfo {
  CIRGenFunction &CGF;
  QualType AtomicTy;
  QualType ValueTy;
  uint64_t AtomicSizeInBits;
  uint64_t ValueSizeInBits;
  CharUnits AtomicAlign;
  CharUnits ValueAlign;
  TypeEvaluationKind EvaluationKind;
  bool UseLibcall;
  LValue LVal;
  CIRGenBitFieldInfo BFI;
  mlir::Location loc;

public:
  AtomicInfo(CIRGenFunction &CGF, LValue &lvalue, mlir::Location l)
      : CGF(CGF), AtomicSizeInBits(0), ValueSizeInBits(0),
        EvaluationKind(TEK_Scalar), UseLibcall(true), loc(l) {
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
  TypeEvaluationKind getEvaluationKind() const { return EvaluationKind; }
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
  mlir::Value convertRValueToInt(RValue RVal) const;

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
                            LVal.getBaseInfo());
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
  bool requiresMemSetZero(llvm::Type *type) const;

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
static Address buildValToTemp(CIRGenFunction &CGF, Expr *E) {
  Address DeclPtr = CGF.CreateMemTemp(
      E->getType(), CGF.getLoc(E->getSourceRange()), ".atomictmp");
  CGF.buildAnyExprToMem(E, DeclPtr, E->getType().getQualifiers(),
                        /*Init*/ true);
  return DeclPtr;
}

Address AtomicInfo::castToAtomicIntPointer(Address addr) const {
  auto intTy = addr.getElementType().dyn_cast<mlir::cir::IntType>();
  // Don't bother with int casts if the integer size is the same.
  if (intTy && intTy.getWidth() == AtomicSizeInBits)
    return addr;
  auto ty = CGF.getBuilder().getUIntNTy(AtomicSizeInBits);
  return addr.withElementType(ty);
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
static mlir::cir::IntAttr getConstOpIntAttr(mlir::Value v) {
  mlir::Operation *op = v.getDefiningOp();
  mlir::cir::IntAttr constVal;
  while (auto c = dyn_cast<mlir::cir::CastOp>(op))
    op = c.getOperand().getDefiningOp();
  if (auto c = dyn_cast<mlir::cir::ConstantOp>(op)) {
    if (c.getType().isa<mlir::cir::IntType>())
      constVal = c.getValue().cast<mlir::cir::IntAttr>();
  }
  return constVal;
}

static void buildAtomicOp(CIRGenFunction &CGF, AtomicExpr *E, Address Dest,
                          Address Ptr, Address Val1, Address Val2,
                          mlir::Value IsWeak, mlir::Value FailureOrder,
                          uint64_t Size, mlir::cir::MemOrder Order,
                          uint8_t Scope) {
  assert(!UnimplementedFeature::syncScopeID());
  StringRef Op;
  [[maybe_unused]] bool PostOpMinMax = false;
  auto loc = CGF.getLoc(E->getSourceRange());

  switch (E->getOp()) {
  case AtomicExpr::AO__c11_atomic_init:
  case AtomicExpr::AO__opencl_atomic_init:
    llvm_unreachable("Already handled!");

  case AtomicExpr::AO__c11_atomic_compare_exchange_strong:
  case AtomicExpr::AO__hip_atomic_compare_exchange_strong:
  case AtomicExpr::AO__opencl_atomic_compare_exchange_strong:
    llvm_unreachable("NYI");
    return;
  case AtomicExpr::AO__c11_atomic_compare_exchange_weak:
  case AtomicExpr::AO__opencl_atomic_compare_exchange_weak:
  case AtomicExpr::AO__hip_atomic_compare_exchange_weak:
    llvm_unreachable("NYI");
    return;
  case AtomicExpr::AO__atomic_compare_exchange:
  case AtomicExpr::AO__atomic_compare_exchange_n:
  case AtomicExpr::AO__scoped_atomic_compare_exchange:
  case AtomicExpr::AO__scoped_atomic_compare_exchange_n: {
    llvm_unreachable("NYI");
    return;
  }
  case AtomicExpr::AO__c11_atomic_load:
  case AtomicExpr::AO__opencl_atomic_load:
  case AtomicExpr::AO__hip_atomic_load:
  case AtomicExpr::AO__atomic_load_n:
  case AtomicExpr::AO__atomic_load:
  case AtomicExpr::AO__scoped_atomic_load_n:
  case AtomicExpr::AO__scoped_atomic_load: {
    llvm_unreachable("NYI");
    return;
  }

  case AtomicExpr::AO__c11_atomic_store:
  case AtomicExpr::AO__opencl_atomic_store:
  case AtomicExpr::AO__hip_atomic_store:
  case AtomicExpr::AO__atomic_store:
  case AtomicExpr::AO__atomic_store_n:
  case AtomicExpr::AO__scoped_atomic_store:
  case AtomicExpr::AO__scoped_atomic_store_n: {
    llvm_unreachable("NYI");
    return;
  }

  case AtomicExpr::AO__c11_atomic_exchange:
  case AtomicExpr::AO__hip_atomic_exchange:
  case AtomicExpr::AO__opencl_atomic_exchange:
  case AtomicExpr::AO__atomic_exchange_n:
  case AtomicExpr::AO__atomic_exchange:
  case AtomicExpr::AO__scoped_atomic_exchange_n:
  case AtomicExpr::AO__scoped_atomic_exchange:
    llvm_unreachable("NYI");
    break;

  case AtomicExpr::AO__atomic_add_fetch:
  case AtomicExpr::AO__scoped_atomic_add_fetch:
    // In LLVM codegen, the post operation codegen is tracked here.
    [[fallthrough]];
  case AtomicExpr::AO__c11_atomic_fetch_add:
  case AtomicExpr::AO__hip_atomic_fetch_add:
  case AtomicExpr::AO__opencl_atomic_fetch_add:
  case AtomicExpr::AO__atomic_fetch_add:
  case AtomicExpr::AO__scoped_atomic_fetch_add:
    Op = mlir::cir::AtomicAddFetch::getOperationName();
    break;

  case AtomicExpr::AO__atomic_sub_fetch:
  case AtomicExpr::AO__scoped_atomic_sub_fetch:
    // In LLVM codegen, the post operation codegen is tracked here.
    llvm_unreachable("NYI");
    [[fallthrough]];
  case AtomicExpr::AO__c11_atomic_fetch_sub:
  case AtomicExpr::AO__hip_atomic_fetch_sub:
  case AtomicExpr::AO__opencl_atomic_fetch_sub:
  case AtomicExpr::AO__atomic_fetch_sub:
  case AtomicExpr::AO__scoped_atomic_fetch_sub:
    llvm_unreachable("NYI");
    break;

  case AtomicExpr::AO__atomic_min_fetch:
  case AtomicExpr::AO__scoped_atomic_min_fetch:
    PostOpMinMax = true;
    [[fallthrough]];
  case AtomicExpr::AO__c11_atomic_fetch_min:
  case AtomicExpr::AO__hip_atomic_fetch_min:
  case AtomicExpr::AO__opencl_atomic_fetch_min:
  case AtomicExpr::AO__atomic_fetch_min:
  case AtomicExpr::AO__scoped_atomic_fetch_min:
    llvm_unreachable("NYI");
    break;

  case AtomicExpr::AO__atomic_max_fetch:
  case AtomicExpr::AO__scoped_atomic_max_fetch:
    PostOpMinMax = true;
    [[fallthrough]];
  case AtomicExpr::AO__c11_atomic_fetch_max:
  case AtomicExpr::AO__hip_atomic_fetch_max:
  case AtomicExpr::AO__opencl_atomic_fetch_max:
  case AtomicExpr::AO__atomic_fetch_max:
  case AtomicExpr::AO__scoped_atomic_fetch_max:
    llvm_unreachable("NYI");
    break;

  case AtomicExpr::AO__atomic_and_fetch:
  case AtomicExpr::AO__scoped_atomic_and_fetch:
    // In LLVM codegen, the post operation codegen is tracked here.
    llvm_unreachable("NYI");
    [[fallthrough]];
  case AtomicExpr::AO__c11_atomic_fetch_and:
  case AtomicExpr::AO__hip_atomic_fetch_and:
  case AtomicExpr::AO__opencl_atomic_fetch_and:
  case AtomicExpr::AO__atomic_fetch_and:
  case AtomicExpr::AO__scoped_atomic_fetch_and:
    llvm_unreachable("NYI");
    break;

  case AtomicExpr::AO__atomic_or_fetch:
  case AtomicExpr::AO__scoped_atomic_or_fetch:
    // In LLVM codegen, the post operation codegen is tracked here.
    llvm_unreachable("NYI");
    [[fallthrough]];
  case AtomicExpr::AO__c11_atomic_fetch_or:
  case AtomicExpr::AO__hip_atomic_fetch_or:
  case AtomicExpr::AO__opencl_atomic_fetch_or:
  case AtomicExpr::AO__atomic_fetch_or:
  case AtomicExpr::AO__scoped_atomic_fetch_or:
    llvm_unreachable("NYI");
    break;

  case AtomicExpr::AO__atomic_xor_fetch:
  case AtomicExpr::AO__scoped_atomic_xor_fetch:
    // In LLVM codegen, the post operation codegen is tracked here.
    llvm_unreachable("NYI");
    [[fallthrough]];
  case AtomicExpr::AO__c11_atomic_fetch_xor:
  case AtomicExpr::AO__hip_atomic_fetch_xor:
  case AtomicExpr::AO__opencl_atomic_fetch_xor:
  case AtomicExpr::AO__atomic_fetch_xor:
  case AtomicExpr::AO__scoped_atomic_fetch_xor:
    llvm_unreachable("NYI");
    break;

  case AtomicExpr::AO__atomic_nand_fetch:
  case AtomicExpr::AO__scoped_atomic_nand_fetch:
    // In LLVM codegen, the post operation codegen is tracked here.
    llvm_unreachable("NYI");
    [[fallthrough]];
  case AtomicExpr::AO__c11_atomic_fetch_nand:
  case AtomicExpr::AO__atomic_fetch_nand:
  case AtomicExpr::AO__scoped_atomic_fetch_nand:
    llvm_unreachable("NYI");
    break;
  }

  assert(Op.size() && "expected operation name to build");
  auto &builder = CGF.getBuilder();

  auto LoadVal1 = builder.createLoad(loc, Val1);

  SmallVector<mlir::Value> atomicOperands = {Ptr.getPointer(), LoadVal1};
  SmallVector<mlir::Type> atomicResTys = {
      Ptr.getPointer().getType().cast<mlir::cir::PointerType>().getPointee()};
  auto orderAttr = mlir::cir::MemOrderAttr::get(builder.getContext(), Order);
  auto RMWI = builder.create(loc, builder.getStringAttr(Op), atomicOperands,
                             atomicResTys, {});
  RMWI->setAttr("mem_order", orderAttr);
  if (E->isVolatile())
    RMWI->setAttr("is_volatile", mlir::UnitAttr::get(builder.getContext()));
  auto Result = RMWI->getResult(0);

  if (PostOpMinMax)
    llvm_unreachable("NYI");

  // This should be handled in LowerToLLVM.cpp, still tracking here for now.
  if (E->getOp() == AtomicExpr::AO__atomic_nand_fetch ||
      E->getOp() == AtomicExpr::AO__scoped_atomic_nand_fetch)
    llvm_unreachable("NYI");

  builder.createStore(loc, Result, Dest);
}

static void buildAtomicOp(CIRGenFunction &CGF, AtomicExpr *Expr, Address Dest,
                          Address Ptr, Address Val1, Address Val2,
                          mlir::Value IsWeak, mlir::Value FailureOrder,
                          uint64_t Size, mlir::cir::MemOrder Order,
                          mlir::Value Scope) {
  auto ScopeModel = Expr->getScopeModel();

  // LLVM atomic instructions always have synch scope. If clang atomic
  // expression has no scope operand, use default LLVM synch scope.
  if (!ScopeModel) {
    assert(!UnimplementedFeature::syncScopeID());
    buildAtomicOp(CGF, Expr, Dest, Ptr, Val1, Val2, IsWeak, FailureOrder, Size,
                  Order, /*FIXME(cir): LLVM default scope*/ 1);
    return;
  }

  // Handle constant scope.
  if (getConstOpIntAttr(Scope)) {
    assert(!UnimplementedFeature::syncScopeID());
    llvm_unreachable("NYI");
    return;
  }

  // Handle non-constant scope.
  llvm_unreachable("NYI");
}

RValue CIRGenFunction::buildAtomicExpr(AtomicExpr *E) {
  QualType AtomicTy = E->getPtr()->getType()->getPointeeType();
  QualType MemTy = AtomicTy;
  if (const AtomicType *AT = AtomicTy->getAs<AtomicType>())
    MemTy = AT->getValueType();
  mlir::Value IsWeak = nullptr, OrderFail = nullptr;

  Address Val1 = Address::invalid();
  Address Val2 = Address::invalid();
  Address Dest = Address::invalid();
  Address Ptr = buildPointerWithAlignment(E->getPtr());

  if (E->getOp() == AtomicExpr::AO__c11_atomic_init ||
      E->getOp() == AtomicExpr::AO__opencl_atomic_init) {
    llvm_unreachable("NYI");
  }

  auto TInfo = getContext().getTypeInfoInChars(AtomicTy);
  uint64_t Size = TInfo.Width.getQuantity();
  unsigned MaxInlineWidthInBits = getTarget().getMaxAtomicInlineWidth();

  bool Oversized = getContext().toBits(TInfo.Width) > MaxInlineWidthInBits;
  bool Misaligned = (Ptr.getAlignment() % TInfo.Width) != 0;
  bool UseLibcall = Misaligned | Oversized;
  bool ShouldCastToIntPtrTy = true;

  CharUnits MaxInlineWidth =
      getContext().toCharUnitsFromBits(MaxInlineWidthInBits);

  DiagnosticsEngine &Diags = CGM.getDiags();

  if (Misaligned) {
    Diags.Report(E->getBeginLoc(), diag::warn_atomic_op_misaligned)
        << (int)TInfo.Width.getQuantity()
        << (int)Ptr.getAlignment().getQuantity();
  }

  if (Oversized) {
    Diags.Report(E->getBeginLoc(), diag::warn_atomic_op_oversized)
        << (int)TInfo.Width.getQuantity() << (int)MaxInlineWidth.getQuantity();
  }

  auto Order = buildScalarExpr(E->getOrder());
  auto Scope = E->getScopeModel() ? buildScalarExpr(E->getScope()) : nullptr;

  switch (E->getOp()) {
  case AtomicExpr::AO__c11_atomic_init:
  case AtomicExpr::AO__opencl_atomic_init:
    llvm_unreachable("Already handled above with EmitAtomicInit!");

  case AtomicExpr::AO__atomic_load_n:
  case AtomicExpr::AO__scoped_atomic_load_n:
  case AtomicExpr::AO__c11_atomic_load:
  case AtomicExpr::AO__opencl_atomic_load:
  case AtomicExpr::AO__hip_atomic_load:
    break;

  case AtomicExpr::AO__atomic_load:
  case AtomicExpr::AO__scoped_atomic_load:
    llvm_unreachable("NYI");
    break;

  case AtomicExpr::AO__atomic_store:
  case AtomicExpr::AO__scoped_atomic_store:
    llvm_unreachable("NYI");
    break;

  case AtomicExpr::AO__atomic_exchange:
  case AtomicExpr::AO__scoped_atomic_exchange:
    llvm_unreachable("NYI");
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
    llvm_unreachable("NYI");
    break;

  case AtomicExpr::AO__c11_atomic_fetch_add:
  case AtomicExpr::AO__c11_atomic_fetch_sub:
  case AtomicExpr::AO__hip_atomic_fetch_add:
  case AtomicExpr::AO__hip_atomic_fetch_sub:
  case AtomicExpr::AO__opencl_atomic_fetch_add:
  case AtomicExpr::AO__opencl_atomic_fetch_sub:
    llvm_unreachable("NYI");
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
    Val1 = buildValToTemp(*this, E->getVal1());
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
    llvm_unreachable("NYI");
  } else if (E->isCmpXChg())
    llvm_unreachable("NYI");
  else if (!RValTy->isVoidType()) {
    Dest = Atomics.CreateTempAlloca();
    if (ShouldCastToIntPtrTy)
      Dest = Atomics.castToAtomicIntPointer(Dest);
  }

  // Use a library call.  See: http://gcc.gnu.org/wiki/Atomic/GCCMM/LIbrary .
  if (UseLibcall) {
    bool UseOptimizedLibcall = false;
    switch (E->getOp()) {
    case AtomicExpr::AO__c11_atomic_init:
    case AtomicExpr::AO__opencl_atomic_init:
      llvm_unreachable("Already handled above with EmitAtomicInit!");

    case AtomicExpr::AO__atomic_fetch_add:
    case AtomicExpr::AO__atomic_fetch_and:
    case AtomicExpr::AO__atomic_fetch_max:
    case AtomicExpr::AO__atomic_fetch_min:
    case AtomicExpr::AO__atomic_fetch_nand:
    case AtomicExpr::AO__atomic_fetch_or:
    case AtomicExpr::AO__atomic_fetch_sub:
    case AtomicExpr::AO__atomic_fetch_xor:
    case AtomicExpr::AO__atomic_add_fetch:
    case AtomicExpr::AO__atomic_and_fetch:
    case AtomicExpr::AO__atomic_max_fetch:
    case AtomicExpr::AO__atomic_min_fetch:
    case AtomicExpr::AO__atomic_nand_fetch:
    case AtomicExpr::AO__atomic_or_fetch:
    case AtomicExpr::AO__atomic_sub_fetch:
    case AtomicExpr::AO__atomic_xor_fetch:
    case AtomicExpr::AO__c11_atomic_fetch_add:
    case AtomicExpr::AO__c11_atomic_fetch_and:
    case AtomicExpr::AO__c11_atomic_fetch_max:
    case AtomicExpr::AO__c11_atomic_fetch_min:
    case AtomicExpr::AO__c11_atomic_fetch_nand:
    case AtomicExpr::AO__c11_atomic_fetch_or:
    case AtomicExpr::AO__c11_atomic_fetch_sub:
    case AtomicExpr::AO__c11_atomic_fetch_xor:
    case AtomicExpr::AO__hip_atomic_fetch_add:
    case AtomicExpr::AO__hip_atomic_fetch_and:
    case AtomicExpr::AO__hip_atomic_fetch_max:
    case AtomicExpr::AO__hip_atomic_fetch_min:
    case AtomicExpr::AO__hip_atomic_fetch_or:
    case AtomicExpr::AO__hip_atomic_fetch_sub:
    case AtomicExpr::AO__hip_atomic_fetch_xor:
    case AtomicExpr::AO__opencl_atomic_fetch_add:
    case AtomicExpr::AO__opencl_atomic_fetch_and:
    case AtomicExpr::AO__opencl_atomic_fetch_max:
    case AtomicExpr::AO__opencl_atomic_fetch_min:
    case AtomicExpr::AO__opencl_atomic_fetch_or:
    case AtomicExpr::AO__opencl_atomic_fetch_sub:
    case AtomicExpr::AO__opencl_atomic_fetch_xor:
    case AtomicExpr::AO__scoped_atomic_fetch_add:
    case AtomicExpr::AO__scoped_atomic_fetch_and:
    case AtomicExpr::AO__scoped_atomic_fetch_max:
    case AtomicExpr::AO__scoped_atomic_fetch_min:
    case AtomicExpr::AO__scoped_atomic_fetch_nand:
    case AtomicExpr::AO__scoped_atomic_fetch_or:
    case AtomicExpr::AO__scoped_atomic_fetch_sub:
    case AtomicExpr::AO__scoped_atomic_fetch_xor:
    case AtomicExpr::AO__scoped_atomic_add_fetch:
    case AtomicExpr::AO__scoped_atomic_and_fetch:
    case AtomicExpr::AO__scoped_atomic_max_fetch:
    case AtomicExpr::AO__scoped_atomic_min_fetch:
    case AtomicExpr::AO__scoped_atomic_nand_fetch:
    case AtomicExpr::AO__scoped_atomic_or_fetch:
    case AtomicExpr::AO__scoped_atomic_sub_fetch:
    case AtomicExpr::AO__scoped_atomic_xor_fetch:
      // For these, only library calls for certain sizes exist.
      UseOptimizedLibcall = true;
      break;

    case AtomicExpr::AO__atomic_load:
    case AtomicExpr::AO__atomic_store:
    case AtomicExpr::AO__atomic_exchange:
    case AtomicExpr::AO__atomic_compare_exchange:
    case AtomicExpr::AO__scoped_atomic_load:
    case AtomicExpr::AO__scoped_atomic_store:
    case AtomicExpr::AO__scoped_atomic_exchange:
    case AtomicExpr::AO__scoped_atomic_compare_exchange:
      // Use the generic version if we don't know that the operand will be
      // suitably aligned for the optimized version.
      if (Misaligned)
        break;
      [[fallthrough]];
    case AtomicExpr::AO__atomic_load_n:
    case AtomicExpr::AO__atomic_store_n:
    case AtomicExpr::AO__atomic_exchange_n:
    case AtomicExpr::AO__atomic_compare_exchange_n:
    case AtomicExpr::AO__c11_atomic_load:
    case AtomicExpr::AO__c11_atomic_store:
    case AtomicExpr::AO__c11_atomic_exchange:
    case AtomicExpr::AO__c11_atomic_compare_exchange_weak:
    case AtomicExpr::AO__c11_atomic_compare_exchange_strong:
    case AtomicExpr::AO__hip_atomic_load:
    case AtomicExpr::AO__hip_atomic_store:
    case AtomicExpr::AO__hip_atomic_exchange:
    case AtomicExpr::AO__hip_atomic_compare_exchange_weak:
    case AtomicExpr::AO__hip_atomic_compare_exchange_strong:
    case AtomicExpr::AO__opencl_atomic_load:
    case AtomicExpr::AO__opencl_atomic_store:
    case AtomicExpr::AO__opencl_atomic_exchange:
    case AtomicExpr::AO__opencl_atomic_compare_exchange_weak:
    case AtomicExpr::AO__opencl_atomic_compare_exchange_strong:
    case AtomicExpr::AO__scoped_atomic_load_n:
    case AtomicExpr::AO__scoped_atomic_store_n:
    case AtomicExpr::AO__scoped_atomic_exchange_n:
    case AtomicExpr::AO__scoped_atomic_compare_exchange_n:
      // Only use optimized library calls for sizes for which they exist.
      // FIXME: Size == 16 optimized library functions exist too.
      if (Size == 1 || Size == 2 || Size == 4 || Size == 8)
        UseOptimizedLibcall = true;
      break;
    }

    CallArgList Args;
    if (!UseOptimizedLibcall) {
      llvm_unreachable("NYI");
    }
    // TODO(cir): Atomic address is the first or second parameter
    // The OpenCL atomic library functions only accept pointer arguments to
    // generic address space.
    llvm_unreachable("NYI");

    std::string LibCallName;
    [[maybe_unused]] QualType LoweredMemTy =
        MemTy->isPointerType() ? getContext().getIntPtrType() : MemTy;
    QualType RetTy;
    [[maybe_unused]] bool HaveRetTy = false;
    [[maybe_unused]] bool PostOpMinMax = false;
    switch (E->getOp()) {
    case AtomicExpr::AO__c11_atomic_init:
    case AtomicExpr::AO__opencl_atomic_init:
      llvm_unreachable("Already handled!");

    // There is only one libcall for compare an exchange, because there is no
    // optimisation benefit possible from a libcall version of a weak compare
    // and exchange.
    // bool __atomic_compare_exchange(size_t size, void *mem, void *expected,
    //                                void *desired, int success, int failure)
    // bool __atomic_compare_exchange_N(T *mem, T *expected, T desired,
    //                                  int success, int failure)
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
    // T __atomic_exchange_N(T *mem, T val, int order)
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
    // void __atomic_store_N(T *mem, T val, int order)
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
    // T __atomic_load_N(T *mem, int order)
    case AtomicExpr::AO__atomic_load:
    case AtomicExpr::AO__atomic_load_n:
    case AtomicExpr::AO__c11_atomic_load:
    case AtomicExpr::AO__hip_atomic_load:
    case AtomicExpr::AO__opencl_atomic_load:
    case AtomicExpr::AO__scoped_atomic_load:
    case AtomicExpr::AO__scoped_atomic_load_n:
      LibCallName = "__atomic_load";
      llvm_unreachable("NYI");
      break;
    // T __atomic_add_fetch_N(T *mem, T val, int order)
    // T __atomic_fetch_add_N(T *mem, T val, int order)
    case AtomicExpr::AO__atomic_add_fetch:
    case AtomicExpr::AO__scoped_atomic_add_fetch:
      llvm_unreachable("NYI");
      [[fallthrough]];
    case AtomicExpr::AO__atomic_fetch_add:
    case AtomicExpr::AO__c11_atomic_fetch_add:
    case AtomicExpr::AO__hip_atomic_fetch_add:
    case AtomicExpr::AO__opencl_atomic_fetch_add:
    case AtomicExpr::AO__scoped_atomic_fetch_add:
      LibCallName = "__atomic_fetch_add";
      llvm_unreachable("NYI");
      break;
    // T __atomic_and_fetch_N(T *mem, T val, int order)
    // T __atomic_fetch_and_N(T *mem, T val, int order)
    case AtomicExpr::AO__atomic_and_fetch:
    case AtomicExpr::AO__scoped_atomic_and_fetch:
      llvm_unreachable("NYI");
      [[fallthrough]];
    case AtomicExpr::AO__atomic_fetch_and:
    case AtomicExpr::AO__c11_atomic_fetch_and:
    case AtomicExpr::AO__hip_atomic_fetch_and:
    case AtomicExpr::AO__opencl_atomic_fetch_and:
    case AtomicExpr::AO__scoped_atomic_fetch_and:
      LibCallName = "__atomic_fetch_and";
      llvm_unreachable("NYI");
      break;
    // T __atomic_or_fetch_N(T *mem, T val, int order)
    // T __atomic_fetch_or_N(T *mem, T val, int order)
    case AtomicExpr::AO__atomic_or_fetch:
    case AtomicExpr::AO__scoped_atomic_or_fetch:
      llvm_unreachable("NYI");
      [[fallthrough]];
    case AtomicExpr::AO__atomic_fetch_or:
    case AtomicExpr::AO__c11_atomic_fetch_or:
    case AtomicExpr::AO__hip_atomic_fetch_or:
    case AtomicExpr::AO__opencl_atomic_fetch_or:
    case AtomicExpr::AO__scoped_atomic_fetch_or:
      LibCallName = "__atomic_fetch_or";
      llvm_unreachable("NYI");
      break;
    // T __atomic_sub_fetch_N(T *mem, T val, int order)
    // T __atomic_fetch_sub_N(T *mem, T val, int order)
    case AtomicExpr::AO__atomic_sub_fetch:
    case AtomicExpr::AO__scoped_atomic_sub_fetch:
      llvm_unreachable("NYI");
      [[fallthrough]];
    case AtomicExpr::AO__atomic_fetch_sub:
    case AtomicExpr::AO__c11_atomic_fetch_sub:
    case AtomicExpr::AO__hip_atomic_fetch_sub:
    case AtomicExpr::AO__opencl_atomic_fetch_sub:
    case AtomicExpr::AO__scoped_atomic_fetch_sub:
      LibCallName = "__atomic_fetch_sub";
      llvm_unreachable("NYI");
      break;
    // T __atomic_xor_fetch_N(T *mem, T val, int order)
    // T __atomic_fetch_xor_N(T *mem, T val, int order)
    case AtomicExpr::AO__atomic_xor_fetch:
    case AtomicExpr::AO__scoped_atomic_xor_fetch:
      llvm_unreachable("NYI");
      [[fallthrough]];
    case AtomicExpr::AO__atomic_fetch_xor:
    case AtomicExpr::AO__c11_atomic_fetch_xor:
    case AtomicExpr::AO__hip_atomic_fetch_xor:
    case AtomicExpr::AO__opencl_atomic_fetch_xor:
    case AtomicExpr::AO__scoped_atomic_fetch_xor:
      LibCallName = "__atomic_fetch_xor";
      llvm_unreachable("NYI");
      break;
    case AtomicExpr::AO__atomic_min_fetch:
    case AtomicExpr::AO__scoped_atomic_min_fetch:
      llvm_unreachable("NYI");
      [[fallthrough]];
    case AtomicExpr::AO__atomic_fetch_min:
    case AtomicExpr::AO__c11_atomic_fetch_min:
    case AtomicExpr::AO__scoped_atomic_fetch_min:
    case AtomicExpr::AO__hip_atomic_fetch_min:
    case AtomicExpr::AO__opencl_atomic_fetch_min:
      LibCallName = E->getValueType()->isSignedIntegerType()
                        ? "__atomic_fetch_min"
                        : "__atomic_fetch_umin";
      llvm_unreachable("NYI");
      break;
    case AtomicExpr::AO__atomic_max_fetch:
    case AtomicExpr::AO__scoped_atomic_max_fetch:
      llvm_unreachable("NYI");
      [[fallthrough]];
    case AtomicExpr::AO__atomic_fetch_max:
    case AtomicExpr::AO__c11_atomic_fetch_max:
    case AtomicExpr::AO__hip_atomic_fetch_max:
    case AtomicExpr::AO__opencl_atomic_fetch_max:
    case AtomicExpr::AO__scoped_atomic_fetch_max:
      LibCallName = E->getValueType()->isSignedIntegerType()
                        ? "__atomic_fetch_max"
                        : "__atomic_fetch_umax";
      llvm_unreachable("NYI");
      break;
    // T __atomic_nand_fetch_N(T *mem, T val, int order)
    // T __atomic_fetch_nand_N(T *mem, T val, int order)
    case AtomicExpr::AO__atomic_nand_fetch:
    case AtomicExpr::AO__scoped_atomic_nand_fetch:
      llvm_unreachable("NYI");
      [[fallthrough]];
    case AtomicExpr::AO__atomic_fetch_nand:
    case AtomicExpr::AO__c11_atomic_fetch_nand:
    case AtomicExpr::AO__scoped_atomic_fetch_nand:
      LibCallName = "__atomic_fetch_nand";
      llvm_unreachable("NYI");
      break;
    }

    if (E->isOpenCL()) {
      LibCallName =
          std::string("__opencl") + StringRef(LibCallName).drop_front(1).str();
    }
    // Optimized functions have the size in their name.
    if (UseOptimizedLibcall) {
      llvm_unreachable("NYI");
    }
    // By default, assume we return a value of the atomic type.
    llvm_unreachable("NYI");
  }

  [[maybe_unused]] bool IsStore =
      E->getOp() == AtomicExpr::AO__c11_atomic_store ||
      E->getOp() == AtomicExpr::AO__opencl_atomic_store ||
      E->getOp() == AtomicExpr::AO__hip_atomic_store ||
      E->getOp() == AtomicExpr::AO__atomic_store ||
      E->getOp() == AtomicExpr::AO__atomic_store_n ||
      E->getOp() == AtomicExpr::AO__scoped_atomic_store ||
      E->getOp() == AtomicExpr::AO__scoped_atomic_store_n;
  [[maybe_unused]] bool IsLoad =
      E->getOp() == AtomicExpr::AO__c11_atomic_load ||
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
    if (mlir::cir::isValidCIRAtomicOrderingCABI(ord)) {
      switch ((mlir::cir::MemOrder)ord) {
      case mlir::cir::MemOrder::Relaxed:
        buildAtomicOp(*this, E, Dest, Ptr, Val1, Val2, IsWeak, OrderFail, Size,
                      mlir::cir::MemOrder::Relaxed, Scope);
        break;
      case mlir::cir::MemOrder::Consume:
      case mlir::cir::MemOrder::Acquire:
        if (IsStore)
          break; // Avoid crashing on code with undefined behavior
        buildAtomicOp(*this, E, Dest, Ptr, Val1, Val2, IsWeak, OrderFail, Size,
                      mlir::cir::MemOrder::Acquire, Scope);
        break;
      case mlir::cir::MemOrder::Release:
        if (IsLoad)
          break; // Avoid crashing on code with undefined behavior
        buildAtomicOp(*this, E, Dest, Ptr, Val1, Val2, IsWeak, OrderFail, Size,
                      mlir::cir::MemOrder::Release, Scope);
        break;
      case mlir::cir::MemOrder::AcquireRelease:
        if (IsLoad || IsStore)
          break; // Avoid crashing on code with undefined behavior
        buildAtomicOp(*this, E, Dest, Ptr, Val1, Val2, IsWeak, OrderFail, Size,
                      mlir::cir::MemOrder::AcquireRelease, Scope);
        break;
      case mlir::cir::MemOrder::SequentiallyConsistent:
        buildAtomicOp(*this, E, Dest, Ptr, Val1, Val2, IsWeak, OrderFail, Size,
                      mlir::cir::MemOrder::SequentiallyConsistent, Scope);
        break;
      }
    }
    if (RValTy->isVoidType()) {
      llvm_unreachable("NYI");
    }

    return convertTempToRValue(Dest.withElementType(convertTypeForMem(RValTy)),
                               RValTy, E->getExprLoc());
  }

  // Long case, when Order isn't obviously constant.
  llvm_unreachable("NYI");
}