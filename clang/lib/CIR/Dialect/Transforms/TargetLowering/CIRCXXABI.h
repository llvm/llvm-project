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

#include "mlir/Transforms/DialectConversion.h"
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

public:
  virtual ~CIRCXXABI();

  /// Lower the given data member pointer type to its ABI type. The returned
  /// type is also a CIR type.
  virtual mlir::Type
  lowerDataMemberType(cir::DataMemberType type,
                      const mlir::TypeConverter &typeConverter) const = 0;

  /// Lower the given member function pointer type to its ABI type. The returned
  /// type is also a CIR type.
  virtual mlir::Type
  lowerMethodType(cir::MethodType type,
                  const mlir::TypeConverter &typeConverter) const = 0;

  /// Lower the given data member pointer constant to a constant of the ABI
  /// type. The returned constant is represented as an attribute as well.
  virtual mlir::TypedAttr
  lowerDataMemberConstant(cir::DataMemberAttr attr,
                          const mlir::DataLayout &layout,
                          const mlir::TypeConverter &typeConverter) const = 0;

  /// Lower the given member function pointer constant to a constant of the ABI
  /// type. The returned constant is represented as an attribute as well.
  virtual mlir::TypedAttr
  lowerMethodConstant(cir::MethodAttr attr, const mlir::DataLayout &layout,
                      const mlir::TypeConverter &typeConverter) const = 0;

  /// Lower the given cir.get_runtime_member op to a sequence of more
  /// "primitive" CIR operations that act on the ABI types.
  virtual mlir::Operation *
  lowerGetRuntimeMember(cir::GetRuntimeMemberOp op, mlir::Type loweredResultTy,
                        mlir::Value loweredAddr, mlir::Value loweredMember,
                        mlir::OpBuilder &builder) const = 0;

  /// Lower the given cir.get_method op to a sequence of more "primitive" CIR
  /// operations that act on the ABI types. The lowered result values will be
  /// stored in the given loweredResults array.
  virtual void
  lowerGetMethod(cir::GetMethodOp op, mlir::Value &callee, mlir::Value &thisArg,
                 mlir::Value loweredMethod, mlir::Value loweredObjectPtr,
                 mlir::ConversionPatternRewriter &rewriter) const = 0;

  /// Lower the given cir.base_data_member op to a sequence of more "primitive"
  /// CIR operations that act on the ABI types.
  virtual mlir::Value lowerBaseDataMember(cir::BaseDataMemberOp op,
                                          mlir::Value loweredSrc,
                                          mlir::OpBuilder &builder) const = 0;

  /// Lower the given cir.derived_data_member op to a sequence of more
  /// "primitive" CIR operations that act on the ABI types.
  virtual mlir::Value
  lowerDerivedDataMember(cir::DerivedDataMemberOp op, mlir::Value loweredSrc,
                         mlir::OpBuilder &builder) const = 0;

  virtual mlir::Value lowerDataMemberCmp(cir::CmpOp op, mlir::Value loweredLhs,
                                         mlir::Value loweredRhs,
                                         mlir::OpBuilder &builder) const = 0;

  virtual mlir::Value lowerMethodCmp(cir::CmpOp op, mlir::Value loweredLhs,
                                     mlir::Value loweredRhs,
                                     mlir::OpBuilder &builder) const = 0;
};

/// Creates an Itanium-family ABI.
std::unique_ptr<CIRCXXABI> createItaniumCXXABI(LowerModule &lm);

} // namespace cir

#endif // CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_CIRCXXABI_H
