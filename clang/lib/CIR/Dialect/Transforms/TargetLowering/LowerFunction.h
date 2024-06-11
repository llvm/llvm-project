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

using CallArgList = SmallVector<Value, 8>;

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

  /// Rewrite a call operation to abide to the ABI calling convention.
  LogicalResult rewriteCallOp(CallOp op,
                              ReturnValueSlot retValSlot = ReturnValueSlot());
  Value rewriteCallOp(FuncType calleeTy, FuncOp origCallee, CallOp callOp,
                      ReturnValueSlot retValSlot, Value Chain = nullptr);
  Value rewriteCallOp(const LowerFunctionInfo &CallInfo, FuncOp Callee,
                      CallOp Caller, ReturnValueSlot ReturnValue,
                      CallArgList &CallArgs, CallOp CallOrInvoke,
                      bool isMustTail, Location loc);

  /// Get an appropriate 'undef' value for the given type.
  Value getUndefRValue(Type Ty);
};

} // namespace cir
} // namespace mlir

#endif // LLVM_CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_LOWERFUNCTION_H
