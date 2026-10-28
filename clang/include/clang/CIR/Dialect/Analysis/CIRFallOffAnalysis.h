//===- CIRFallOffAnalysis.h - Can control flow off the end of a function --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares an analysis that classifies whether control can reach the
// end of a function body, the CIR counterpart of the CFG-based fall-off check
// behind -Wreturn-type. It runs on CIRGen output, where the terminator emitted
// for flowing off the end of a body is marked `implicit`.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_DIALECT_ANALYSIS_CIRFALLOFFANALYSIS_H
#define CLANG_CIR_DIALECT_ANALYSIS_CIRFALLOFFANALYSIS_H

#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
class ModuleOp;
} // namespace mlir

namespace cir {

/// How control can reach the end of a function body. Mirrors the
/// classification used by Sema's CFG-based fall-off check.
enum class FallOffKind {
  /// The function was not analyzed (e.g. it has no body).
  Unknown,
  /// Control never reaches the end of the body, but a return is reachable.
  Never,
  /// Control never reaches the end of the body and no return is reachable.
  NeverAndNoReturn,
  /// Control reaches the end of the body on some paths only.
  Maybe,
  /// Control reaches the end of the body on every path that completes.
  Always,
};

llvm::StringRef stringifyFallOffKind(FallOffKind kind);

/// Classify whether control can flow off the end of `func`'s body.
FallOffKind analyzeFallOff(cir::FuncOp func);

/// A cheap, conservative check of whether control may reach the end of
/// `func`. When false, analyzeFallOff(func) is never or never-and-no-return.
bool mayFallOff(cir::FuncOp func);

struct FallOffResult {
  cir::FuncOp func;
  FallOffKind kind = FallOffKind::Unknown;
};

/// analyzeFallOff over the definitions `filter` accepts (all if null), in
/// parallel when allowed: `filter` may run concurrently. In module order.
llvm::SmallVector<FallOffResult>
analyzeFallOff(mlir::ModuleOp module,
               llvm::function_ref<bool(cir::FuncOp)> filter = nullptr);

} // namespace cir

#endif // CLANG_CIR_DIALECT_ANALYSIS_CIRFALLOFFANALYSIS_H
