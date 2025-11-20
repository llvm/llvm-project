//===--- FallThroughWarning.h - CIR Fall-Through Analysis ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Defines the FallThroughWarningPass for CIR fall-through analysis.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CIR_SEMA_FALLTHROUGHWARNING_H
#define LLVM_CLANG_CIR_SEMA_FALLTHROUGHWARNING_H

#include "clang/AST/Type.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Sema/AnalysisBasedWarnings.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
namespace cir {
class FuncOp;
} // namespace cir

namespace clang {
class Sema;

enum ControlFlowKind {
  UnknownFallThrough,
  NeverFallThrough,
  MaybeFallThrough,
  AlwaysFallThrough,
  NeverFallThroughOrReturn
};

/// Configuration for fall-through diagnostics
struct CheckFallThroughDiagnostics {
  unsigned diagFallThrough = 0;
  unsigned diagReturn = 0;
  unsigned diagFallThroughAttr = 0;
  unsigned funKind = 0;
  SourceLocation funcLoc;

  bool checkDiagnostics(DiagnosticsEngine &d, bool returnsVoid,
                        bool hasNoReturn) const;
};

/// Pass for analyzing fall-through behavior in CIR functions
class FallThroughWarningPass {
public:
  FallThroughWarningPass() = default;

  /// Check fall-through behavior for a CIR function body
  void checkFallThroughForFuncBody(Sema &s, cir::FuncOp cfg, QualType blockType,
                                   const CheckFallThroughDiagnostics &cd);
  ControlFlowKind checkFallThrough(cir::FuncOp cfg);
  mlir::DenseSet<mlir::Block *> getLiveSet(cir::FuncOp cfg);
};

} // namespace clang

#endif // LLVM_CLANG_CIR_SEMA_FALLTHROUGHWARNING_H
