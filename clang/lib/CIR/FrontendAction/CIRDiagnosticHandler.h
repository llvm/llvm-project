//===--- CIRDiagnosticHandler.h - Route MLIR diags to clang ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_LIB_CIR_FRONTENDACTION_CIRDIAGNOSTICHANDLER_H
#define CLANG_LIB_CIR_FRONTENDACTION_CIRDIAGNOSTICHANDLER_H

#include "mlir/IR/Diagnostics.h"

namespace clang {
class DiagnosticsEngine;
class FileManager;
class SourceLocation;
class SourceManager;
} // namespace clang

namespace mlir {
class Location;
class MLIRContext;
} // namespace mlir

namespace cir {

/// Routes MLIR-side diagnostics emitted during CIR passes, the verifier, and
/// CIR-to-LLVM lowering through the surrounding clang::DiagnosticsEngine.
///
/// MLIR locations carrying file/line/column info are translated to a
/// clang::SourceLocation via the active SourceManager + FileManager so errors
/// surface in the user's preferred format. Locations that cannot be translated
/// fall back to an invalid clang::SourceLocation.
class CIRDiagnosticHandler : public mlir::ScopedDiagnosticHandler {
public:
  CIRDiagnosticHandler(mlir::MLIRContext *ctx, clang::DiagnosticsEngine &diags,
                       clang::SourceManager &srcMgr,
                       clang::FileManager &fileMgr);

private:
  mlir::LogicalResult handle(mlir::Diagnostic &diag);
  void emit(mlir::Diagnostic &diag, bool isNote);
  clang::SourceLocation translateLoc(mlir::Location loc);

  clang::DiagnosticsEngine &Diags;
  clang::SourceManager &SrcMgr;
  clang::FileManager &FileMgr;
};

} // namespace cir

#endif // CLANG_LIB_CIR_FRONTENDACTION_CIRDIAGNOSTICHANDLER_H
