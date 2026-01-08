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
#include "llvm/Support/ErrorHandling.h"

namespace cir {

namespace {

class LowerItaniumCXXABI : public CIRCXXABI {
public:
  LowerItaniumCXXABI(LowerModule &lm) : CIRCXXABI(lm) {}

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

  mlir::Operation *
  lowerGetRuntimeMember(cir::GetRuntimeMemberOp op, mlir::Type loweredResultTy,
                        mlir::Value loweredAddr, mlir::Value loweredMember,
                        mlir::OpBuilder &builder) const override;
};

} // namespace

std::unique_ptr<CIRCXXABI> createItaniumCXXABI(LowerModule &lm) {
  return std::make_unique<LowerItaniumCXXABI>(lm);
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

} // namespace cir
