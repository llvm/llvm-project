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

#include "mlir/IR/Value.h"
#include "clang/CIR/Dialect/Builder/CIRBaseBuilder.h"
#include "clang/CIR/Dialect/IR/CIRDataLayout.h"

namespace mlir {
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
};

/// Creates an Itanium-family ABI.
CIRCXXABI *CreateItaniumCXXABI(LowerModule &CGM);

} // namespace cir
} // namespace mlir

// FIXME(cir): Merge this into the CIRCXXABI class above. To do so, this code
// should be updated to follow some level of codegen parity.
namespace cir {

enum class AArch64ABIKind {
  AAPCS = 0,
  DarwinPCS,
  Win64,
  AAPCSSoft,
};

class LoweringPrepareCXXABI {
public:
  static LoweringPrepareCXXABI *createItaniumABI();
  static LoweringPrepareCXXABI *createAArch64ABI(AArch64ABIKind k);

  virtual mlir::Value lowerVAArg(CIRBaseBuilderTy &builder,
                                 mlir::cir::VAArgOp op,
                                 const cir::CIRDataLayout &datalayout) = 0;
  virtual ~LoweringPrepareCXXABI() {}

  virtual mlir::Value lowerDynamicCast(CIRBaseBuilderTy &builder,
                                       clang::ASTContext &astCtx,
                                       mlir::cir::DynamicCastOp op) = 0;
};
} // namespace cir

#endif // LLVM_CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_CIRCXXABI_H
