//===----- CIRCXXABI.h - Interface to C++ ABIs for CIR Dialect --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file partially mimics the CodeGen/CGCXXABI.h class. The main difference
// is that this is adapted to operate on the CIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_CIRCXXABI_H
#define CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_CIRCXXABI_H

#include "aiir/Transforms/DialectConversion.h"
#include "clang/AST/CharUnits.h"
#include "clang/CIR/Dialect/Builder/CIRBaseBuilder.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"

namespace cir {

// Forward declarations.
class LowerModule;

class CIRCXXABI {
  friend class LowerModule;

protected:
  LowerModule &lm;

  CIRCXXABI(LowerModule &lm) : lm(lm) {}

  unsigned getPtrSizeInBits() const;

public:
  virtual ~CIRCXXABI();

  /// Lower the given data member pointer type to its ABI type. The returned
  /// type is also a CIR type.
  virtual aiir::Type
  lowerDataMemberType(cir::DataMemberType type,
                      const aiir::TypeConverter &typeConverter) const = 0;

  /// Lower the given member function pointer type to its ABI type. The returned
  /// type is also a CIR type.
  virtual aiir::Type
  lowerMethodType(cir::MethodType type,
                  const aiir::TypeConverter &typeConverter) const = 0;

  /// Lower the given data member pointer constant to a constant of the ABI
  /// type. The returned constant is represented as an attribute as well.
  virtual aiir::TypedAttr
  lowerDataMemberConstant(cir::DataMemberAttr attr,
                          const aiir::DataLayout &layout,
                          const aiir::TypeConverter &typeConverter) const = 0;

  /// Lower the given member function pointer constant to a constant of the ABI
  /// type. The returned constant is represented as an attribute as well.
  virtual aiir::TypedAttr
  lowerMethodConstant(cir::MethodAttr attr, const aiir::DataLayout &layout,
                      const aiir::TypeConverter &typeConverter) const = 0;

  /// Lower the given cir.get_runtime_member op to a sequence of more
  /// "primitive" CIR operations that act on the ABI types.
  virtual aiir::Operation *
  lowerGetRuntimeMember(cir::GetRuntimeMemberOp op, aiir::Type loweredResultTy,
                        aiir::Value loweredAddr, aiir::Value loweredMember,
                        aiir::OpBuilder &builder) const = 0;

  /// Lower the given cir.get_method op to a sequence of more "primitive" CIR
  /// operations that act on the ABI types. The lowered result values will be
  /// stored in the given loweredResults array.
  virtual void
  lowerGetMethod(cir::GetMethodOp op, aiir::Value &callee, aiir::Value &thisArg,
                 aiir::Value loweredMethod, aiir::Value loweredObjectPtr,
                 aiir::ConversionPatternRewriter &rewriter) const = 0;

  /// Lower the given cir.base_data_member op to a sequence of more "primitive"
  /// CIR operations that act on the ABI types.
  virtual aiir::Value lowerBaseDataMember(cir::BaseDataMemberOp op,
                                          aiir::Value loweredSrc,
                                          aiir::OpBuilder &builder) const = 0;

  /// Lower the given cir.derived_data_member op to a sequence of more
  /// "primitive" CIR operations that act on the ABI types.
  virtual aiir::Value
  lowerDerivedDataMember(cir::DerivedDataMemberOp op, aiir::Value loweredSrc,
                         aiir::OpBuilder &builder) const = 0;

  virtual aiir::Value lowerBaseMethod(cir::BaseMethodOp op,
                                      aiir::Value loweredSrc,
                                      aiir::OpBuilder &builder) const = 0;

  virtual aiir::Value lowerDerivedMethod(cir::DerivedMethodOp op,
                                         aiir::Value loweredSrc,
                                         aiir::OpBuilder &builder) const = 0;

  virtual aiir::Value lowerDataMemberCmp(cir::CmpOp op, aiir::Value loweredLhs,
                                         aiir::Value loweredRhs,
                                         aiir::OpBuilder &builder) const = 0;

  virtual aiir::Value lowerMethodCmp(cir::CmpOp op, aiir::Value loweredLhs,
                                     aiir::Value loweredRhs,
                                     aiir::OpBuilder &builder) const = 0;

  virtual aiir::Value
  lowerDataMemberBitcast(cir::CastOp op, aiir::Type loweredDstTy,
                         aiir::Value loweredSrc,
                         aiir::OpBuilder &builder) const = 0;

  virtual aiir::Value
  lowerDataMemberToBoolCast(cir::CastOp op, aiir::Value loweredSrc,
                            aiir::OpBuilder &builder) const = 0;

  virtual aiir::Value lowerMethodBitcast(cir::CastOp op,
                                         aiir::Type loweredDstTy,
                                         aiir::Value loweredSrc,
                                         aiir::OpBuilder &builder) const = 0;

  virtual aiir::Value lowerMethodToBoolCast(cir::CastOp op,
                                            aiir::Value loweredSrc,
                                            aiir::OpBuilder &builder) const = 0;

  virtual aiir::Value lowerDynamicCast(cir::DynamicCastOp op,
                                       aiir::OpBuilder &builder) const = 0;

  virtual aiir::Value
  lowerVTableGetTypeInfo(cir::VTableGetTypeInfoOp op,
                         aiir::OpBuilder &builder) const = 0;

  /// Read the array cookie for a dynamically-allocated array whose first
  /// element is at \p elementPtr. Returns the number of elements, the
  /// original allocation pointer (before the cookie) as a void*, and the
  /// cookie size in bytes. Delegates to getArrayCookieSizeImpl and
  /// readArrayCookieImpl.
  void readArrayCookie(aiir::Location loc, aiir::Value elementPtr,
                       const aiir::DataLayout &dataLayout,
                       CIRBaseBuilderTy &builder, aiir::Value &numElements,
                       aiir::Value &allocPtr,
                       clang::CharUnits &cookieSize) const;

protected:
  /// Returns the cookie size in bytes for a dynamically-allocated array of
  /// elements with the given type. Only called when a cookie is required.
  virtual clang::CharUnits
  getArrayCookieSizeImpl(aiir::Type elementType,
                         const aiir::DataLayout &dataLayout) const = 0;

  /// Reads the element count from an array cookie. \p allocPtr is a byte
  /// pointer to the start of the allocation (the beginning of the cookie).
  /// \p cookieSize is the value returned by getArrayCookieSizeImpl.
  /// \p cookieAlignment is the alignment at the cookie start, derived from
  /// the element type's ABI alignment.
  virtual aiir::Value readArrayCookieImpl(aiir::Location loc,
                                          aiir::Value allocPtr,
                                          clang::CharUnits cookieSize,
                                          clang::CharUnits cookieAlignment,
                                          const aiir::DataLayout &dataLayout,
                                          CIRBaseBuilderTy &builder) const = 0;
};

/// Creates an Itanium-family ABI.
std::unique_ptr<CIRCXXABI> createItaniumCXXABI(LowerModule &lm);

} // namespace cir

#endif // CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_CIRCXXABI_H
