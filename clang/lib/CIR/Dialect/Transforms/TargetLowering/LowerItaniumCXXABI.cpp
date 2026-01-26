//===---- LowerItaniumCXXABI.cpp - Emit CIR code Itanium-specific code  ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This provides CIR lowering logic targeting the Itanium C++ ABI. The class in
// this file generates records that follow the Itanium C++ ABI, which is
// documented at:
//  https://itanium-cxx-abi.github.io/cxx-abi/abi.html
//  https://itanium-cxx-abi.github.io/cxx-abi/abi-eh.html
//
// It also supports the closely-related ARM ABI, documented at:
// https://developer.arm.com/documentation/ihi0041/g/
//
// This file partially mimics clang/lib/CodeGen/ItaniumCXXABI.cpp. The queries
// are adapted to operate on the CIR dialect, however.
//
//===----------------------------------------------------------------------===//

#include "CIRCXXABI.h"
#include "LowerModule.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/Support/ErrorHandling.h"

namespace cir {

namespace {

class LowerItaniumCXXABI : public CIRCXXABI {
protected:
  bool useARMMethodPtrABI;

public:
  LowerItaniumCXXABI(LowerModule &lm, bool useARMMethodPtrABI = false)
      : CIRCXXABI(lm), useARMMethodPtrABI(useARMMethodPtrABI) {}

  /// Lower the given data member pointer type to its ABI type. The returned
  /// type is also a CIR type.
  virtual mlir::Type
  lowerDataMemberType(cir::DataMemberType type,
                      const mlir::TypeConverter &typeConverter) const override;

  mlir::Type
  lowerMethodType(cir::MethodType type,
                  const mlir::TypeConverter &typeConverter) const override;

  mlir::TypedAttr lowerDataMemberConstant(
      cir::DataMemberAttr attr, const mlir::DataLayout &layout,
      const mlir::TypeConverter &typeConverter) const override;

  mlir::TypedAttr
  lowerMethodConstant(cir::MethodAttr attr, const mlir::DataLayout &layout,
                      const mlir::TypeConverter &typeConverter) const override;

  mlir::Operation *
  lowerGetRuntimeMember(cir::GetRuntimeMemberOp op, mlir::Type loweredResultTy,
                        mlir::Value loweredAddr, mlir::Value loweredMember,
                        mlir::OpBuilder &builder) const override;

  void lowerGetMethod(cir::GetMethodOp op, mlir::Value &callee,
                      mlir::Value &thisArg, mlir::Value loweredMethod,
                      mlir::Value loweredObjectPtr,
                      mlir::ConversionPatternRewriter &rewriter) const override;

  mlir::Value lowerBaseDataMember(cir::BaseDataMemberOp op,
                                  mlir::Value loweredSrc,
                                  mlir::OpBuilder &builder) const override;

  mlir::Value lowerDerivedDataMember(cir::DerivedDataMemberOp op,
                                     mlir::Value loweredSrc,
                                     mlir::OpBuilder &builder) const override;

  mlir::Value lowerDataMemberCmp(cir::CmpOp op, mlir::Value loweredLhs,
                                 mlir::Value loweredRhs,
                                 mlir::OpBuilder &builder) const override;

  mlir::Value lowerMethodCmp(cir::CmpOp op, mlir::Value loweredLhs,
                             mlir::Value loweredRhs,
                             mlir::OpBuilder &builder) const override;

  mlir::Value lowerDataMemberBitcast(cir::CastOp op, mlir::Type loweredDstTy,
                                     mlir::Value loweredSrc,
                                     mlir::OpBuilder &builder) const override;

  mlir::Value
  lowerDataMemberToBoolCast(cir::CastOp op, mlir::Value loweredSrc,
                            mlir::OpBuilder &builder) const override;

  mlir::Value lowerMethodBitcast(cir::CastOp op, mlir::Type loweredDstTy,
                                 mlir::Value loweredSrc,
                                 mlir::OpBuilder &builder) const override;

  mlir::Value lowerMethodToBoolCast(cir::CastOp op, mlir::Value loweredSrc,
                                    mlir::OpBuilder &builder) const override;

  mlir::Value lowerDynamicCast(cir::DynamicCastOp op,
                               mlir::OpBuilder &builder) const override;
};

} // namespace

std::unique_ptr<CIRCXXABI> createItaniumCXXABI(LowerModule &lm) {
  switch (lm.getCXXABIKind()) {
  // Note that AArch64 uses the generic ItaniumCXXABI class since it doesn't
  // include the other 32-bit ARM oddities: constructor/destructor return values
  // and array cookies.
  case clang::TargetCXXABI::GenericAArch64:
  case clang::TargetCXXABI::AppleARM64:
    // TODO: this isn't quite right, clang uses AppleARM64CXXABI which inherits
    // from ARMCXXABI. We'll have to follow suit.
    assert(!cir::MissingFeatures::appleArm64CXXABI());
    return std::make_unique<LowerItaniumCXXABI>(lm,
                                                /*useARMMethodPtrABI=*/true);

  case clang::TargetCXXABI::GenericItanium:
    return std::make_unique<LowerItaniumCXXABI>(lm);

  case clang::TargetCXXABI::Microsoft:
    llvm_unreachable("Microsoft ABI is not Itanium-based");
  default:
    llvm_unreachable("Other Itanium ABI?");
  }
}

static cir::IntType getPtrDiffCIRTy(LowerModule &lm) {
  const clang::TargetInfo &target = lm.getTarget();
  clang::TargetInfo::IntType ptrdiffTy =
      target.getPtrDiffType(clang::LangAS::Default);
  return cir::IntType::get(lm.getMLIRContext(), target.getTypeWidth(ptrdiffTy),
                           target.isTypeSigned(ptrdiffTy));
}

mlir::Type LowerItaniumCXXABI::lowerDataMemberType(
    cir::DataMemberType type, const mlir::TypeConverter &typeConverter) const {
  // Itanium C++ ABI 2.3.1:
  //   A data member pointer is represented as the data member's offset in bytes
  //   from the address point of an object of the base type, as a ptrdiff_t.
  return getPtrDiffCIRTy(lm);
}

mlir::Type LowerItaniumCXXABI::lowerMethodType(
    cir::MethodType type, const mlir::TypeConverter &typeConverter) const {
  // Itanium C++ ABI 2.3.2:
  //    In all representations, the basic ABI properties of member function
  //    pointer types are those of the following class, where fnptr_t is the
  //    appropriate function-pointer type for a member function of this type:
  //
  //    struct {
  //      fnptr_t ptr;
  //      ptrdiff_t adj;
  //    };

  cir::IntType ptrdiffCIRTy = getPtrDiffCIRTy(lm);

  // Note that clang CodeGen emits struct{ptrdiff_t, ptrdiff_t} for member
  // function pointers. Let's follow this approach.
  return cir::RecordType::get(type.getContext(), {ptrdiffCIRTy, ptrdiffCIRTy},
                              /*packed=*/false, /*padded=*/false,
                              cir::RecordType::Struct);
}

mlir::TypedAttr LowerItaniumCXXABI::lowerDataMemberConstant(
    cir::DataMemberAttr attr, const mlir::DataLayout &layout,
    const mlir::TypeConverter &typeConverter) const {
  uint64_t memberOffset;
  if (attr.isNullPtr()) {
    // Itanium C++ ABI 2.3:
    //   A NULL pointer is represented as -1.
    memberOffset = -1ull;
  } else {
    // Itanium C++ ABI 2.3:
    //   A pointer to data member is an offset from the base address of
    //   the class object containing it, represented as a ptrdiff_t
    unsigned memberIndex = attr.getMemberIndex().value();
    memberOffset =
        attr.getType().getClassTy().getElementOffset(layout, memberIndex);
  }

  mlir::Type abiTy = lowerDataMemberType(attr.getType(), typeConverter);
  return cir::IntAttr::get(abiTy, memberOffset);
}

mlir::TypedAttr LowerItaniumCXXABI::lowerMethodConstant(
    cir::MethodAttr attr, const mlir::DataLayout &layout,
    const mlir::TypeConverter &typeConverter) const {
  cir::IntType ptrdiffCIRTy = getPtrDiffCIRTy(lm);

  // lowerMethodType returns the CIR type used to represent the method pointer
  // in an ABI-specific way. That's why lowerMethodType returns cir::RecordType
  // here.
  auto loweredMethodTy = mlir::cast<cir::RecordType>(
      lowerMethodType(attr.getType(), typeConverter));

  auto zero = cir::IntAttr::get(ptrdiffCIRTy, 0);

  // Itanium C++ ABI 2.3.2:
  //   In all representations, the basic ABI properties of member function
  //   pointer types are those of the following class, where fnptr_t is the
  //   appropriate function-pointer type for a member function of this type:
  //
  //   struct {
  //     fnptr_t ptr;
  //     ptrdiff_t adj;
  //   };

  if (attr.isNull()) {
    // Itanium C++ ABI 2.3.2:
    //
    //   In the standard representation, a null member function pointer is
    //   represented with ptr set to a null pointer. The value of adj is
    //   unspecified for null member function pointers.
    //
    // clang CodeGen emits struct{null, null} for null member function pointers.
    // Let's do the same here.
    return cir::ConstRecordAttr::get(
        loweredMethodTy, mlir::ArrayAttr::get(attr.getContext(), {zero, zero}));
  }

  if (attr.isVirtual()) {
    if (useARMMethodPtrABI) {
      // ARM C++ ABI 3.2.1:
      //   This ABI specifies that adj contains twice the this
      //   adjustment, plus 1 if the member function is virtual. The
      //   least significant bit of adj then makes exactly the same
      //   discrimination as the least significant bit of ptr does for
      //   Itanium.
      llvm_unreachable("ARM method ptr abi NYI");
    }

    // Itanium C++ ABI 2.3.2:
    //
    //   In the standard representation, a member function pointer for a
    //   virtual function is represented with ptr set to 1 plus the function's
    //   v-table entry offset (in bytes), converted to a function pointer as if
    //   by reinterpret_cast<fnptr_t>(uintfnptr_t(1 + offset)), where
    //   uintfnptr_t is an unsigned integer of the same size as fnptr_t.
    auto ptr =
        cir::IntAttr::get(ptrdiffCIRTy, 1 + attr.getVtableOffset().value());
    return cir::ConstRecordAttr::get(
        loweredMethodTy, mlir::ArrayAttr::get(attr.getContext(), {ptr, zero}));
  }

  // Itanium C++ ABI 2.3.2:
  //
  //   A member function pointer for a non-virtual member function is
  //   represented with ptr set to a pointer to the function, using the base
  //   ABI's representation of function pointers.
  auto ptr = cir::GlobalViewAttr::get(ptrdiffCIRTy, attr.getSymbol().value());
  return cir::ConstRecordAttr::get(
      loweredMethodTy, mlir::ArrayAttr::get(attr.getContext(), {ptr, zero}));
}

mlir::Operation *LowerItaniumCXXABI::lowerGetRuntimeMember(
    cir::GetRuntimeMemberOp op, mlir::Type loweredResultTy,
    mlir::Value loweredAddr, mlir::Value loweredMember,
    mlir::OpBuilder &builder) const {
  auto byteTy = cir::IntType::get(op.getContext(), 8, true);
  auto bytePtrTy = cir::PointerType::get(
      byteTy,
      mlir::cast<cir::PointerType>(op.getAddr().getType()).getAddrSpace());
  auto objectBytesPtr = cir::CastOp::create(
      builder, op.getLoc(), bytePtrTy, cir::CastKind::bitcast, op.getAddr());
  auto memberBytesPtr = cir::PtrStrideOp::create(
      builder, op.getLoc(), bytePtrTy, objectBytesPtr, loweredMember);
  return cir::CastOp::create(builder, op.getLoc(), op.getType(),
                             cir::CastKind::bitcast, memberBytesPtr);
}

void LowerItaniumCXXABI::lowerGetMethod(
    cir::GetMethodOp op, mlir::Value &callee, mlir::Value &thisArg,
    mlir::Value loweredMethod, mlir::Value loweredObjectPtr,
    mlir::ConversionPatternRewriter &rewriter) const {
  // In the Itanium and ARM ABIs, method pointers have the form:
  //   struct { ptrdiff_t ptr; ptrdiff_t adj; } memptr;
  //
  // In the Itanium ABI:
  //  - method pointers are virtual if (memptr.ptr & 1) is nonzero
  //  - the this-adjustment is (memptr.adj)
  //  - the virtual offset is (memptr.ptr - 1)
  //
  // In the ARM ABI:
  //  - method pointers are virtual if (memptr.adj & 1) is nonzero
  //  - the this-adjustment is (memptr.adj >> 1)
  //  - the virtual offset is (memptr.ptr)
  // ARM uses 'adj' for the virtual flag because Thumb functions
  // may be only single-byte aligned.
  //
  // If the member is virtual, the adjusted 'this' pointer points
  // to a vtable pointer from which the virtual offset is applied.
  //
  // If the member is non-virtual, memptr.ptr is the address of
  // the function to call.

  mlir::ImplicitLocOpBuilder locBuilder(op.getLoc(), rewriter);
  mlir::Type calleePtrTy = op.getCallee().getType();

  cir::IntType ptrdiffCIRTy = getPtrDiffCIRTy(lm);
  mlir::Value ptrdiffOne =
      cir::ConstantOp::create(locBuilder, cir::IntAttr::get(ptrdiffCIRTy, 1));

  mlir::Value adj =
      cir::ExtractMemberOp::create(locBuilder, ptrdiffCIRTy, loweredMethod, 1);
  if (useARMMethodPtrABI) {
    op.emitError("ARM method ptr abi NYI");
    return;
  }

  // Apply the adjustment to the 'this' pointer.
  mlir::Type thisVoidPtrTy =
      cir::PointerType::get(cir::VoidType::get(locBuilder.getContext()),
                            op.getObject().getType().getAddrSpace());
  mlir::Value thisVoidPtr = cir::CastOp::create(
      locBuilder, thisVoidPtrTy, cir::CastKind::bitcast, loweredObjectPtr);
  thisArg =
      cir::PtrStrideOp::create(locBuilder, thisVoidPtrTy, thisVoidPtr, adj);

  // Load the "ptr" field of the member function pointer and determine if it
  // points to a virtual function.
  mlir::Value methodPtrField =
      cir::ExtractMemberOp::create(locBuilder, ptrdiffCIRTy, loweredMethod, 0);
  mlir::Value virtualBit = cir::BinOp::create(
      rewriter, op.getLoc(), cir::BinOpKind::And, methodPtrField, ptrdiffOne);
  mlir::Value isVirtual;
  if (useARMMethodPtrABI)
    llvm_unreachable("ARM method ptr abi NYI");
  else
    isVirtual = cir::CmpOp::create(locBuilder, cir::CmpOpKind::eq, virtualBit,
                                   ptrdiffOne);

  assert(!cir::MissingFeatures::emitCFICheck());
  assert(!cir::MissingFeatures::emitVFEInfo());
  assert(!cir::MissingFeatures::emitWPDInfo());

  auto buildVirtualCallee = [&](mlir::OpBuilder &b, mlir::Location loc) {
    // Load vtable pointer.
    // Note that vtable pointer always point to the global address space.
    auto vtablePtrTy =
        cir::PointerType::get(cir::IntType::get(b.getContext(), 8, true));
    auto vtablePtrPtrTy = cir::PointerType::get(
        vtablePtrTy, op.getObject().getType().getAddrSpace());
    auto vtablePtrPtr = cir::CastOp::create(b, loc, vtablePtrPtrTy,
                                            cir::CastKind::bitcast, thisArg);
    assert(!cir::MissingFeatures::opTBAA());
    mlir::Value vtablePtr =
        cir::LoadOp::create(b, loc, vtablePtrPtr, /*isDeref=*/false,
                            /*isVolatile=*/false,
                            /*alignment=*/mlir::IntegerAttr(),
                            /*sync_scope=*/cir::SyncScopeKindAttr{},
                            /*mem_order=*/cir::MemOrderAttr());

    // Get the vtable offset.
    mlir::Value vtableOffset = methodPtrField;
    assert(!useARMMethodPtrABI && "ARM method ptr abi NYI");
    vtableOffset = cir::BinOp::create(b, loc, cir::BinOpKind::Sub, vtableOffset,
                                      ptrdiffOne);

    assert(!cir::MissingFeatures::emitCFICheck());
    assert(!cir::MissingFeatures::emitVFEInfo());
    assert(!cir::MissingFeatures::emitWPDInfo());

    // Apply the offset to the vtable pointer and get the pointer to the target
    // virtual function. Then load that pointer to get the callee.
    mlir::Value vfpAddr = cir::PtrStrideOp::create(locBuilder, vtablePtrTy,
                                                   vtablePtr, vtableOffset);
    auto vfpPtrTy = cir::PointerType::get(calleePtrTy);
    mlir::Value vfpPtr = cir::CastOp::create(locBuilder, vfpPtrTy,
                                             cir::CastKind::bitcast, vfpAddr);
    auto fnPtr = cir::LoadOp::create(b, loc, vfpPtr,
                                     /*isDeref=*/false, /*isVolatile=*/false,
                                     /*alignment=*/mlir::IntegerAttr(),
                                     /*sync_scope=*/cir::SyncScopeKindAttr{},
                                     /*mem_order=*/cir::MemOrderAttr());

    cir::YieldOp::create(b, loc, fnPtr.getResult());
    assert(!cir::MissingFeatures::emitCFICheck());
  };

  callee = cir::TernaryOp::create(
               locBuilder, isVirtual, /*thenBuilder=*/buildVirtualCallee,
               /*elseBuilder=*/
               [&](mlir::OpBuilder &b, mlir::Location loc) {
                 auto fnPtr = cir::CastOp::create(b, loc, calleePtrTy,
                                                  cir::CastKind::int_to_ptr,
                                                  methodPtrField);
                 cir::YieldOp::create(b, loc, fnPtr.getResult());
               })
               .getResult();
}

static mlir::Value lowerDataMemberCast(mlir::Operation *op,
                                       mlir::Value loweredSrc,
                                       std::int64_t offset,
                                       bool isDerivedToBase,
                                       mlir::OpBuilder &builder) {
  if (offset == 0)
    return loweredSrc;
  mlir::Location loc = op->getLoc();
  mlir::Type ty = loweredSrc.getType();

  auto getConstantInt = [&](int64_t value) -> cir::ConstantOp {
    return cir::ConstantOp::create(builder, loc, cir::IntAttr::get(ty, value));
  };

  cir::ConstantOp nullValue = getConstantInt(-1);
  auto isNull = cir::CmpOp::create(builder, loc, cir::CmpOpKind::eq, loweredSrc,
                                   nullValue);

  cir::ConstantOp offsetValue = getConstantInt(offset);
  auto binOpKind = isDerivedToBase ? cir::BinOpKind::Sub : cir::BinOpKind::Add;
  cir::BinOp adjustedPtr =
      cir::BinOp::create(builder, loc, ty, binOpKind, loweredSrc, offsetValue);
  adjustedPtr.setNoSignedWrap(true);

  return cir::SelectOp::create(builder, loc, ty, isNull, loweredSrc,
                               adjustedPtr);
}

mlir::Value
LowerItaniumCXXABI::lowerBaseDataMember(cir::BaseDataMemberOp op,
                                        mlir::Value loweredSrc,
                                        mlir::OpBuilder &builder) const {
  return lowerDataMemberCast(op, loweredSrc, op.getOffset().getSExtValue(),
                             /*isDerivedToBase=*/true, builder);
}

mlir::Value
LowerItaniumCXXABI::lowerDerivedDataMember(cir::DerivedDataMemberOp op,
                                           mlir::Value loweredSrc,
                                           mlir::OpBuilder &builder) const {
  return lowerDataMemberCast(op, loweredSrc, op.getOffset().getSExtValue(),
                             /*isDerivedToBase=*/false, builder);
}

mlir::Value
LowerItaniumCXXABI::lowerDataMemberCmp(cir::CmpOp op, mlir::Value loweredLhs,
                                       mlir::Value loweredRhs,
                                       mlir::OpBuilder &builder) const {
  return cir::CmpOp::create(builder, op.getLoc(), op.getKind(), loweredLhs,
                            loweredRhs);
}

mlir::Value LowerItaniumCXXABI::lowerMethodCmp(cir::CmpOp op,
                                               mlir::Value loweredLhs,
                                               mlir::Value loweredRhs,
                                               mlir::OpBuilder &builder) const {
  assert(op.getKind() == cir::CmpOpKind::eq ||
         op.getKind() == cir::CmpOpKind::ne);

  mlir::ImplicitLocOpBuilder locBuilder(op.getLoc(), builder);
  cir::IntType ptrdiffCIRTy = getPtrDiffCIRTy(lm);
  mlir::Value ptrdiffZero =
      cir::ConstantOp::create(locBuilder, cir::IntAttr::get(ptrdiffCIRTy, 0));

  mlir::Value lhsPtrField =
      cir::ExtractMemberOp::create(locBuilder, ptrdiffCIRTy, loweredLhs, 0);
  mlir::Value rhsPtrField =
      cir::ExtractMemberOp::create(locBuilder, ptrdiffCIRTy, loweredRhs, 0);
  mlir::Value ptrCmp =
      cir::CmpOp::create(locBuilder, op.getKind(), lhsPtrField, rhsPtrField);
  mlir::Value ptrCmpToNull =
      cir::CmpOp::create(locBuilder, op.getKind(), lhsPtrField, ptrdiffZero);

  mlir::Value lhsAdjField =
      cir::ExtractMemberOp::create(locBuilder, ptrdiffCIRTy, loweredLhs, 1);
  mlir::Value rhsAdjField =
      cir::ExtractMemberOp::create(locBuilder, ptrdiffCIRTy, loweredRhs, 1);
  mlir::Value adjCmp =
      cir::CmpOp::create(locBuilder, op.getKind(), lhsAdjField, rhsAdjField);

  auto create_and = [&](mlir::Value lhs, mlir::Value rhs) {
    return cir::BinOp::create(locBuilder, cir::BinOpKind::And, lhs, rhs);
  };
  auto create_or = [&](mlir::Value lhs, mlir::Value rhs) {
    return cir::BinOp::create(locBuilder, cir::BinOpKind::Or, lhs, rhs);
  };

  mlir::Value result;
  if (op.getKind() == cir::CmpOpKind::eq) {
    // (lhs.ptr == null || lhs.adj == rhs.adj) && lhs.ptr == rhs.ptr
    result = create_and(ptrCmp, create_or(ptrCmpToNull, adjCmp));
  } else {
    // (lhs.ptr != null && lhs.adj != rhs.adj) || lhs.ptr != rhs.ptr
    result = create_or(ptrCmp, create_and(ptrCmpToNull, adjCmp));
  }

  return result;
}

mlir::Value LowerItaniumCXXABI::lowerDataMemberBitcast(
    cir::CastOp op, mlir::Type loweredDstTy, mlir::Value loweredSrc,
    mlir::OpBuilder &builder) const {
  if (loweredSrc.getType() == loweredDstTy)
    return loweredSrc;

  return cir::CastOp::create(builder, op.getLoc(), loweredDstTy,
                             cir::CastKind::bitcast, loweredSrc);
}

mlir::Value LowerItaniumCXXABI::lowerDataMemberToBoolCast(
    cir::CastOp op, mlir::Value loweredSrc, mlir::OpBuilder &builder) const {
  // Itanium C++ ABI 2.3:
  //   A NULL pointer is represented as -1.
  auto nullAttr = cir::IntAttr::get(getPtrDiffCIRTy(lm), -1);
  auto nullValue = cir::ConstantOp::create(builder, op.getLoc(), nullAttr);
  return cir::CmpOp::create(builder, op.getLoc(), cir::CmpOpKind::ne,
                            loweredSrc, nullValue);
}

mlir::Value
LowerItaniumCXXABI::lowerMethodBitcast(cir::CastOp op, mlir::Type loweredDstTy,
                                       mlir::Value loweredSrc,
                                       mlir::OpBuilder &builder) const {
  if (loweredSrc.getType() == loweredDstTy)
    return loweredSrc;

  return loweredSrc;
}

mlir::Value LowerItaniumCXXABI::lowerMethodToBoolCast(
    cir::CastOp op, mlir::Value loweredSrc, mlir::OpBuilder &builder) const {
  // Itanium C++ ABI 2.3.2:
  //
  //   In the standard representation, a null member function pointer is
  //   represented with ptr set to a null pointer. The value of adj is
  //   unspecified for null member function pointers.
  cir::IntType ptrdiffCIRTy = getPtrDiffCIRTy(lm);
  mlir::Value ptrdiffZero = cir::ConstantOp::create(
      builder, op.getLoc(), cir::IntAttr::get(ptrdiffCIRTy, 0));
  mlir::Value ptrField = cir::ExtractMemberOp::create(
      builder, op.getLoc(), ptrdiffCIRTy, loweredSrc, 0);
  return cir::CmpOp::create(builder, op.getLoc(), cir::CmpOpKind::ne, ptrField,
                            ptrdiffZero);
}

static void buildBadCastCall(mlir::OpBuilder &builder, mlir::Location loc,
                             mlir::FlatSymbolRefAttr badCastFuncRef) {
  cir::CallOp::create(builder, loc, badCastFuncRef, /*resType=*/cir::VoidType(),
                      /*operands=*/mlir::ValueRange{});
  // TODO(cir): Set the 'noreturn' attribute on the function.
  assert(!cir::MissingFeatures::opFuncNoReturn());

  cir::UnreachableOp::create(builder, loc);
  builder.clearInsertionPoint();
}

static mlir::Value buildDynamicCastAfterNullCheck(cir::DynamicCastOp op,
                                                  mlir::OpBuilder &builder) {
  mlir::Location loc = op->getLoc();
  mlir::Value srcValue = op.getSrc();
  cir::DynamicCastInfoAttr castInfo = op.getInfo().value();

  // TODO(cir): consider address space
  assert(!cir::MissingFeatures::addressSpace());

  auto voidPtrTy =
      cir::PointerType::get(cir::VoidType::get(builder.getContext()));

  mlir::Value srcPtr = cir::CastOp::create(builder, loc, voidPtrTy,
                                           cir::CastKind::bitcast, srcValue);
  mlir::Value srcRtti =
      cir::ConstantOp::create(builder, loc, castInfo.getSrcRtti());
  mlir::Value destRtti =
      cir::ConstantOp::create(builder, loc, castInfo.getDestRtti());
  mlir::Value offsetHint =
      cir::ConstantOp::create(builder, loc, castInfo.getOffsetHint());

  mlir::FlatSymbolRefAttr dynCastFuncRef = castInfo.getRuntimeFunc();
  mlir::Value dynCastFuncArgs[4] = {srcPtr, srcRtti, destRtti, offsetHint};

  mlir::Value castedPtr = cir::CallOp::create(builder, loc, dynCastFuncRef,
                                              voidPtrTy, dynCastFuncArgs)
                              .getResult();

  assert(mlir::isa<cir::PointerType>(castedPtr.getType()) &&
         "the return value of __dynamic_cast should be a ptr");

  /// C++ [expr.dynamic.cast]p9:
  ///   A failed cast to reference type throws std::bad_cast
  if (op.isRefCast()) {
    // Emit a cir.if that checks the casted value.
    mlir::Value null = cir::ConstantOp::create(
        builder, loc,
        cir::ConstPtrAttr::get(castedPtr.getType(),
                               builder.getI64IntegerAttr(0)));
    mlir::Value castedPtrIsNull =
        cir::CmpOp::create(builder, loc, cir::CmpOpKind::eq, castedPtr, null);
    cir::IfOp::create(builder, loc, castedPtrIsNull, false,
                      [&](mlir::OpBuilder &, mlir::Location) {
                        buildBadCastCall(builder, loc,
                                         castInfo.getBadCastFunc());
                      });
  }

  // Note that castedPtr is a void*. Cast it to a pointer to the destination
  // type before return.
  return cir::CastOp::create(builder, loc, op.getType(), cir::CastKind::bitcast,
                             castedPtr);
}

static mlir::Value buildDynamicCastToVoidAfterNullCheck(
    cir::DynamicCastOp op, cir::LowerModule &lm, mlir::OpBuilder &builder) {
  mlir::Location loc = op.getLoc();
  bool vtableUsesRelativeLayout = op.getRelativeLayout();

  // TODO(cir): consider address space in this function.
  assert(!cir::MissingFeatures::addressSpace());

  mlir::Type vtableElemTy;
  uint64_t vtableElemAlign;
  if (vtableUsesRelativeLayout) {
    vtableElemTy =
        cir::IntType::get(builder.getContext(), 32, /*isSigned=*/true);
    vtableElemAlign = 4;
  } else {
    vtableElemTy = getPtrDiffCIRTy(lm);
    vtableElemAlign = llvm::divideCeil(
        lm.getTarget().getPointerAlign(clang::LangAS::Default), 8);
  }

  mlir::Type vtableElemPtrTy = cir::PointerType::get(vtableElemTy);
  mlir::Type i64Ty = cir::IntType::get(builder.getContext(), /*width=*/64,
                                       /*isSigned=*/true);

  // Access vtable to get the offset from the given object to its containing
  // complete object.
  // TODO: Add a specialized operation to get the object offset?
  auto vptrPtr = cir::VTableGetVPtrOp::create(builder, loc, op.getSrc());
  mlir::Value vptr = cir::LoadOp::create(
      builder, loc, vptrPtr,
      /*isDeref=*/false,
      /*is_volatile=*/false,
      /*alignment=*/builder.getI64IntegerAttr(vtableElemAlign),
      /*sync_scope=*/cir::SyncScopeKindAttr(),
      /*mem_order=*/cir::MemOrderAttr());
  mlir::Value elementPtr = cir::CastOp::create(builder, loc, vtableElemPtrTy,
                                               cir::CastKind::bitcast, vptr);
  mlir::Value minusTwo =
      cir::ConstantOp::create(builder, loc, cir::IntAttr::get(i64Ty, -2));
  mlir::Value offsetToTopSlotPtr = cir::PtrStrideOp::create(
      builder, loc, vtableElemPtrTy, elementPtr, minusTwo);
  mlir::Value offsetToTop = cir::LoadOp::create(
      builder, loc, offsetToTopSlotPtr,
      /*isDeref=*/false,
      /*is_volatile=*/false,
      /*alignment=*/builder.getI64IntegerAttr(vtableElemAlign),
      /*sync_scope=*/cir::SyncScopeKindAttr(),
      /*mem_order=*/cir::MemOrderAttr());

  auto voidPtrTy =
      cir::PointerType::get(cir::VoidType::get(builder.getContext()));

  // Add the offset to the given pointer to get the cast result.
  // Cast the input pointer to a uint8_t* to allow pointer arithmetic.
  mlir::Type u8PtrTy =
      cir::PointerType::get(cir::IntType::get(builder.getContext(), /*width=*/8,
                                              /*isSigned=*/false));
  mlir::Value srcBytePtr = cir::CastOp::create(
      builder, loc, u8PtrTy, cir::CastKind::bitcast, op.getSrc());
  auto dstBytePtr =
      cir::PtrStrideOp::create(builder, loc, u8PtrTy, srcBytePtr, offsetToTop);
  // Cast the result to a void*.
  return cir::CastOp::create(builder, loc, voidPtrTy, cir::CastKind::bitcast,
                             dstBytePtr);
}

mlir::Value
LowerItaniumCXXABI::lowerDynamicCast(cir::DynamicCastOp op,
                                     mlir::OpBuilder &builder) const {
  mlir::Location loc = op->getLoc();
  mlir::Value srcValue = op.getSrc();

  assert(!cir::MissingFeatures::emitTypeCheck());

  if (op.isRefCast())
    return buildDynamicCastAfterNullCheck(op, builder);

  mlir::Value srcValueIsNotNull = cir::CastOp::create(
      builder, loc, cir::BoolType::get(builder.getContext()),
      cir::CastKind::ptr_to_bool, srcValue);
  return cir::TernaryOp::create(
             builder, loc, srcValueIsNotNull,
             [&](mlir::OpBuilder &, mlir::Location) {
               mlir::Value castedValue =
                   op.isCastToVoid()
                       ? buildDynamicCastToVoidAfterNullCheck(op, lm, builder)
                       : buildDynamicCastAfterNullCheck(op, builder);
               cir::YieldOp::create(builder, loc, castedValue);
             },
             [&](mlir::OpBuilder &, mlir::Location) {
               mlir::Value null = cir::ConstantOp::create(
                   builder, loc,
                   cir::ConstPtrAttr::get(op.getType(),
                                          builder.getI64IntegerAttr(0)));
               cir::YieldOp::create(builder, loc, null);
             })
      .getResult();
}

} // namespace cir
