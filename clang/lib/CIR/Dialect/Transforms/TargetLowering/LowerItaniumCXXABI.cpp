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

  assert(!cir::MissingFeatures::virtualMethodAttr());

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

} // namespace cir
