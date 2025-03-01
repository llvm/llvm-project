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

#ifndef LLVM_CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_CIRCXXABI_H
#define LLVM_CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_CIRCXXABI_H

#include "LowerFunctionInfo.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"
#include "clang/CIR/Dialect/Builder/CIRBaseBuilder.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDataLayout.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/Target/AArch64.h"

namespace cir {

// Forward declarations.
class LowerModule;

class CIRCXXABI {
  friend class LowerModule;

protected:
  LowerModule &LM;

  CIRCXXABI(LowerModule &LM) : LM(LM) {}

public:
  virtual ~CIRCXXABI();

  /// If the C++ ABI requires the given type be returned in a particular way,
  /// this method sets RetAI and returns true.
  virtual bool classifyReturnType(LowerFunctionInfo &FI) const = 0;

  /// Specify how one should pass an argument of a record type.
  enum RecordArgABI {
    /// Pass it using the normal C aggregate rules for the ABI, potentially
    /// introducing extra copies and passing some or all of it in registers.
    RAA_Default = 0,

    /// Pass it on the stack using its defined layout.  The argument must be
    /// evaluated directly into the correct stack position in the arguments
    /// area,
    /// and the call machinery must not move it or introduce extra copies.
    RAA_DirectInMemory,

    /// Pass it as a pointer to temporary memory.
    RAA_Indirect
  };

  /// Returns how an argument of the given record type should be passed.
  /// FIXME(cir): This expects a CXXRecordDecl! Not any record type.
  virtual RecordArgABI getRecordArgABI(const StructType RD) const = 0;

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
  lowerGetMethod(cir::GetMethodOp op, mlir::Value (&loweredResults)[2],
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

  /// Lower the given cir.base_method op to a sequence of more "primitive" CIR
  /// operations that act on the ABI types.
  virtual mlir::Value lowerBaseMethod(cir::BaseMethodOp op,
                                      mlir::Value loweredSrc,
                                      mlir::OpBuilder &builder) const = 0;

  /// Lower the given cir.derived_method op to a sequence of more "primitive"
  /// CIR operations that act on the ABI types.
  virtual mlir::Value lowerDerivedMethod(cir::DerivedMethodOp op,
                                         mlir::Value loweredSrc,
                                         mlir::OpBuilder &builder) const = 0;

  virtual mlir::Value lowerDataMemberCmp(cir::CmpOp op, mlir::Value loweredLhs,
                                         mlir::Value loweredRhs,
                                         mlir::OpBuilder &builder) const = 0;

  virtual mlir::Value lowerMethodCmp(cir::CmpOp op, mlir::Value loweredLhs,
                                     mlir::Value loweredRhs,
                                     mlir::OpBuilder &builder) const = 0;

  virtual mlir::Value
  lowerDataMemberBitcast(cir::CastOp op, mlir::Type loweredDstTy,
                         mlir::Value loweredSrc,
                         mlir::OpBuilder &builder) const = 0;

  virtual mlir::Value
  lowerDataMemberToBoolCast(cir::CastOp op, mlir::Value loweredSrc,
                            mlir::OpBuilder &builder) const = 0;

  virtual mlir::Value lowerMethodBitcast(cir::CastOp op,
                                         mlir::Type loweredDstTy,
                                         mlir::Value loweredSrc,
                                         mlir::OpBuilder &builder) const = 0;

  virtual mlir::Value lowerMethodToBoolCast(cir::CastOp op,
                                            mlir::Value loweredSrc,
                                            mlir::OpBuilder &builder) const = 0;
};

/// Creates an Itanium-family ABI.
CIRCXXABI *CreateItaniumCXXABI(LowerModule &CGM);

} // namespace cir

#endif // LLVM_CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_CIRCXXABI_H
