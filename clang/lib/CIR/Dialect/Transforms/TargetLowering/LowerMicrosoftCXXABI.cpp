//===---- LowerMicrosoftCXXABI.cpp - Emit CIR for Microsoft-specific code -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This provides CIR lowering logic targeting the Microsoft C++ ABI.
//
//===----------------------------------------------------------------------===//

#include "CIRCXXABI.h"
#include "LowerModule.h"
#include "llvm/Support/ErrorHandling.h"

namespace cir {

namespace {

class LowerMicrosoftCXXABI : public CIRCXXABI {
public:
  LowerMicrosoftCXXABI(LowerModule &lm) : CIRCXXABI(lm) {}

  mlir::Type
  lowerDataMemberType(cir::DataMemberType type,
                      const mlir::TypeConverter &typeConverter) const override {
    llvm_unreachable("Microsoft ABI member pointer lowering NYI");
  }

  mlir::Type
  lowerMethodType(cir::MethodType type,
                  const mlir::TypeConverter &typeConverter) const override {
    llvm_unreachable("Microsoft ABI method pointer lowering NYI");
  }

  mlir::TypedAttr lowerDataMemberConstant(
      cir::DataMemberAttr attr, const mlir::DataLayout &layout,
      const mlir::TypeConverter &typeConverter) const override {
    llvm_unreachable("Microsoft ABI member pointer lowering NYI");
  }

  mlir::TypedAttr lowerDataMemberOffsetConstant(
      cir::DataMemberOffsetAttr attr, const mlir::DataLayout &layout,
      const mlir::TypeConverter &typeConverter) const override {
    llvm_unreachable("Microsoft ABI member pointer lowering NYI");
  }

  mlir::TypedAttr
  lowerMethodConstant(cir::MethodAttr attr, const mlir::DataLayout &layout,
                      const mlir::TypeConverter &typeConverter) const override {
    llvm_unreachable("Microsoft ABI method pointer lowering NYI");
  }

  mlir::Operation *
  lowerGetRuntimeMember(cir::GetRuntimeMemberOp op, mlir::Type loweredResultTy,
                        mlir::Value loweredAddr, mlir::Value loweredMember,
                        mlir::OpBuilder &builder) const override {
    llvm_unreachable("Microsoft ABI member pointer lowering NYI");
  }

  void
  lowerGetMethod(cir::GetMethodOp op, mlir::Value &callee, mlir::Value &thisArg,
                 mlir::Value loweredMethod, mlir::Value loweredObjectPtr,
                 mlir::ConversionPatternRewriter &rewriter) const override {
    llvm_unreachable("Microsoft ABI method pointer lowering NYI");
  }

  mlir::Value lowerBaseDataMember(cir::BaseDataMemberOp op,
                                  mlir::Value loweredSrc,
                                  mlir::OpBuilder &builder) const override {
    llvm_unreachable("Microsoft ABI member pointer lowering NYI");
  }

  mlir::Value lowerDerivedDataMember(cir::DerivedDataMemberOp op,
                                     mlir::Value loweredSrc,
                                     mlir::OpBuilder &builder) const override {
    llvm_unreachable("Microsoft ABI member pointer lowering NYI");
  }

  mlir::Value lowerBaseMethod(cir::BaseMethodOp op, mlir::Value loweredSrc,
                              mlir::OpBuilder &builder) const override {
    llvm_unreachable("Microsoft ABI method pointer lowering NYI");
  }

  mlir::Value lowerDerivedMethod(cir::DerivedMethodOp op,
                                 mlir::Value loweredSrc,
                                 mlir::OpBuilder &builder) const override {
    llvm_unreachable("Microsoft ABI method pointer lowering NYI");
  }

  mlir::Value lowerDataMemberCmp(cir::CmpOp op, mlir::Value loweredLhs,
                                 mlir::Value loweredRhs,
                                 mlir::OpBuilder &builder) const override {
    llvm_unreachable("Microsoft ABI member pointer lowering NYI");
  }

  mlir::Value lowerMethodCmp(cir::CmpOp op, mlir::Value loweredLhs,
                             mlir::Value loweredRhs,
                             mlir::OpBuilder &builder) const override {
    llvm_unreachable("Microsoft ABI method pointer lowering NYI");
  }

  mlir::Value lowerDataMemberBitcast(cir::CastOp op, mlir::Type loweredDstTy,
                                     mlir::Value loweredSrc,
                                     mlir::OpBuilder &builder) const override {
    llvm_unreachable("Microsoft ABI member pointer lowering NYI");
  }

  mlir::Value
  lowerDataMemberToBoolCast(cir::CastOp op, mlir::Value loweredSrc,
                            mlir::OpBuilder &builder) const override {
    llvm_unreachable("Microsoft ABI member pointer lowering NYI");
  }

  mlir::Value lowerMethodBitcast(cir::CastOp op, mlir::Type loweredDstTy,
                                 mlir::Value loweredSrc,
                                 mlir::OpBuilder &builder) const override {
    llvm_unreachable("Microsoft ABI method pointer lowering NYI");
  }

  mlir::Value lowerMethodToBoolCast(cir::CastOp op, mlir::Value loweredSrc,
                                    mlir::OpBuilder &builder) const override {
    llvm_unreachable("Microsoft ABI method pointer lowering NYI");
  }

  mlir::Value lowerDynamicCast(cir::DynamicCastOp op,
                               mlir::OpBuilder &builder) const override {
    llvm_unreachable("Microsoft ABI dynamic cast lowering NYI");
  }

  mlir::Value lowerVTableGetTypeInfo(cir::VTableGetTypeInfoOp op,
                                     mlir::OpBuilder &builder) const override {
    llvm_unreachable("Microsoft ABI vtable get type info lowering NYI");
  }

  clang::CharUnits
  getArrayCookieSizeImpl(mlir::Type elementType,
                         const mlir::DataLayout &dataLayout) const override {
    llvm_unreachable("Microsoft ABI array cookie lowering NYI");
  }

  mlir::Value readArrayCookieImpl(mlir::Location loc, mlir::Value allocPtr,
                                  clang::CharUnits cookieSize,
                                  clang::CharUnits cookieAlignment,
                                  const mlir::DataLayout &dataLayout,
                                  CIRBaseBuilderTy &builder) const override {
    llvm_unreachable("Microsoft ABI array cookie lowering NYI");
  }
};

} // namespace

std::unique_ptr<CIRCXXABI> createMicrosoftCXXABI(LowerModule &lm) {
  return std::make_unique<LowerMicrosoftCXXABI>(lm);
}

} // namespace cir
