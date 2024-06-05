//===-- LowerFunction.h - Per-Function state for CIR lowering ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This class partially mimics clang/lib/CodeGen/CGFunctionInfo.h. The queries
// are adapted to operate on the CIR dialect, however. And we only copy code
// related to ABI-specific codegen.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_LOWERFUNCTION_H
#define LLVM_CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_LOWERFUNCTION_H

#include "CIRCXXABI.h"
#include "LowerCall.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"

namespace mlir {
namespace cir {

class LowerFunction {
  LowerFunction(const LowerFunction &) = delete;
  void operator=(const LowerFunction &) = delete;

  friend class CIRCXXABI;

  const clang::TargetInfo &Target;

  PatternRewriter &rewriter;
  FuncOp SrcFn;  // Original ABI-agnostic function.
  FuncOp NewFn;  // New ABI-aware function.
  CallOp callOp; // Call operation to be lowered.

public:
  /// Builder for lowering calling convention of a function definition.
  LowerFunction(LowerModule &LM, PatternRewriter &rewriter, FuncOp srcFn,
                FuncOp newFn);

  /// Builder for lowering calling convention of a call operation.
  LowerFunction(LowerModule &LM, PatternRewriter &rewriter, FuncOp srcFn,
                CallOp callOp);

  ~LowerFunction() = default;

  LowerModule &LM; // Per-module state.
};

} // namespace cir
} // namespace mlir

#endif // LLVM_CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_LOWERFUNCTION_H
