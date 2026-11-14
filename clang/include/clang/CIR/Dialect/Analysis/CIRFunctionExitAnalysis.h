//===- CIRFunctionExitAnalysis.h - How control leaves a function ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares an analysis that classifies how control can leave a
// function: through an explicit return, by reaching the end of the body, or
// not at all. It is the CIR counterpart of Sema's CFG-based CheckFallThrough
// behind -Wreturn-type, and runs on CIRGen output, where the terminator emitted
// at the end of a body is marked `implicit`.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_DIALECT_ANALYSIS_CIRFUNCTIONEXITANALYSIS_H
#define CLANG_CIR_DIALECT_ANALYSIS_CIRFUNCTIONEXITANALYSIS_H

#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
class ModuleOp;
} // namespace mlir

namespace cir {

/// How control can leave a function. Mirrors Sema's ControlFlowKind.
enum class FunctionExitKind {
  /// The function was not analyzed (e.g. it has no body).
  Unknown,
  /// Control leaves only through explicit returns; the end of the body is
  /// unreachable.
  ReturnOnly,
  /// Control never leaves: neither a return nor the end of the body is
  /// reachable.
  NoReturn,
  /// Control reaches the end of the body on some paths only.
  MayReachEnd,
  /// Control reaches the end of the body on every path that completes.
  MustReachEnd,
};

llvm::StringRef stringifyFunctionExitKind(FunctionExitKind kind);

/// Classify how control can leave `func`.
FunctionExitKind analyzeFunctionExits(cir::FuncOp func);

/// A cheap, conservative check of whether control may reach the end of
/// `func`'s body. When false, analyzeFunctionExits(func) is ReturnOnly or
/// NoReturn.
bool mayReachEnd(cir::FuncOp func);

struct FunctionExitResult {
  cir::FuncOp func;
  FunctionExitKind kind = FunctionExitKind::Unknown;
};

/// analyzeFunctionExits over the definitions `filter` accepts (all if null), in
/// parallel when allowed: `filter` may run concurrently. In module order.
llvm::SmallVector<FunctionExitResult>
analyzeFunctionExits(mlir::ModuleOp module,
                     llvm::function_ref<bool(cir::FuncOp)> filter = nullptr);

} // namespace cir

#endif // CLANG_CIR_DIALECT_ANALYSIS_CIRFUNCTIONEXITANALYSIS_H
