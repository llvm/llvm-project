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
#include "clang/CIR/Dialect/IR/CIRDialect.h"
namespace cir {
class FuncOp;
} // namespace cir

namespace clang {
class Sema;
Decl *getDeclByName(ASTContext &context, StringRef name);
enum ControlFlowKind {
  AlwaysFallThrough,
  UnknownFallThrough,
  MaybeFallThrough,
  NeverFallThrough,
  NeverFallThroughOrReturn,
  Undetermined,
};

/// Configuration for fall-through diagnostics
struct CheckFallThroughDiagnostics {
  unsigned diagFallThroughHasNoReturn = 0;
  unsigned diagFallThroughReturnsNonVoid = 0;
  unsigned diagNeverFallThroughOrReturn = 0;
  unsigned funKind = 0;
  SourceLocation funcLoc;

  static CheckFallThroughDiagnostics makeForFunction(Sema &s, const Decl *func);
  static CheckFallThroughDiagnostics makeForCoroutine(const Decl *func);
  static CheckFallThroughDiagnostics makeForBlock();
  static CheckFallThroughDiagnostics makeForLambda();
  bool checkDiagnostics(DiagnosticsEngine &d, bool returnsVoid,
                        bool hasNoReturn) const;
};
/// Check if a return operation returns a phony value (uninitialized __retval)
bool isPhonyReturn(cir::ReturnOp returnOp);

/// Pass for analyzing fall-through behavior in CIR functions
class FallThroughWarningPass {
public:
  FallThroughWarningPass() = default;

  /// Check fall-through behavior for a CIR function body
  void checkFallThroughForFuncBody(Sema &s, cir::FuncOp cfg, QualType blockType,
                                   const CheckFallThroughDiagnostics &cd);
  ControlFlowKind checkFallThrough(cir::FuncOp cfg);
  mlir::SetVector<mlir::Block *> getLiveSet(cir::FuncOp cfg);
};

} // namespace clang

#endif // LLVM_CLANG_CIR_SEMA_FALLTHROUGHWARNING_H
