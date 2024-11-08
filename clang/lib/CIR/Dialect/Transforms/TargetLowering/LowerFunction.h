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
#include "clang/CIR/TypeEvaluationKind.h"

namespace cir {

using CallArgList = llvm::SmallVector<mlir::Value, 8>;

class LowerFunction {
  LowerFunction(const LowerFunction &) = delete;
  void operator=(const LowerFunction &) = delete;

  friend class CIRCXXABI;

  const clang::TargetInfo &Target;

  mlir::PatternRewriter &rewriter;
  FuncOp SrcFn;  // Original ABI-agnostic function.
  FuncOp NewFn;  // New ABI-aware function.
  CallOp callOp; // Call operation to be lowered.

public:
  /// Builder for lowering calling convention of a function definition.
  LowerFunction(LowerModule &LM, mlir::PatternRewriter &rewriter, FuncOp srcFn,
                FuncOp newFn);

  /// Builder for lowering calling convention of a call operation.
  LowerFunction(LowerModule &LM, mlir::PatternRewriter &rewriter, FuncOp srcFn,
                CallOp callOp);

  ~LowerFunction() = default;

  LowerModule &LM; // Per-module state.

  mlir::PatternRewriter &getRewriter() const { return rewriter; }

  const clang::TargetInfo &getTarget() const { return Target; }

  // Build ABI/Target-specific function prologue.
  llvm::LogicalResult
  buildFunctionProlog(const LowerFunctionInfo &FI, FuncOp Fn,
                      llvm::MutableArrayRef<mlir::BlockArgument> Args);

  // Build ABI/Target-specific function epilogue.
  llvm::LogicalResult buildFunctionEpilog(const LowerFunctionInfo &FI);

  // Parity with CodeGenFunction::GenerateCode. Keep in mind that several
  // sections in the original function are focused on codegen unrelated to the
  // ABI. Such sections are handled in CIR's codegen, not here.
  llvm::LogicalResult generateCode(FuncOp oldFn, FuncOp newFn,
                                   const LowerFunctionInfo &FnInfo);

  // Emit the most simple cir.store possible (e.g. a store for a whole
  // struct), which can later be broken down in other CIR levels (or prior
  // to dialect codegen).
  void buildAggregateStore(mlir::Value Val, mlir::Value Dest,
                           bool DestIsVolatile);

  // Emit a simple bitcast for a coerced aggregate type to convert it from an
  // ABI-agnostic to an ABI-aware type.
  mlir::Value buildAggregateBitcast(mlir::Value Val, mlir::Type DestTy);

  /// Rewrite a call operation to abide to the ABI calling convention.
  llvm::LogicalResult
  rewriteCallOp(CallOp op, ReturnValueSlot retValSlot = ReturnValueSlot());
  mlir::Value rewriteCallOp(FuncType calleeTy, FuncOp origCallee, CallOp callOp,
                            ReturnValueSlot retValSlot,
                            mlir::Value Chain = nullptr);
  mlir::Value rewriteCallOp(const LowerFunctionInfo &CallInfo, FuncOp Callee,
                            CallOp Caller, ReturnValueSlot ReturnValue,
                            CallArgList &CallArgs, CallOp CallOrInvoke,
                            bool isMustTail, mlir::Location loc);

  /// Get an appropriate 'undef' value for the given type.
  mlir::Value getUndefRValue(mlir::Type Ty);

  /// Return the TypeEvaluationKind of Type \c T.
  static cir::TypeEvaluationKind getEvaluationKind(mlir::Type T);

  static bool hasScalarEvaluationKind(mlir::Type T) {
    return getEvaluationKind(T) == cir::TypeEvaluationKind::TEK_Scalar;
  }
};

} // namespace cir

#endif // LLVM_CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_LOWERFUNCTION_H
