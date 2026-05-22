//===--- CIRDiagnosticHandler.cpp - Route MLIR diags to clang ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CIRDiagnosticHandler.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticFrontend.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"

namespace cir {

CIRDiagnosticHandler::CIRDiagnosticHandler(mlir::MLIRContext *ctx,
                                           clang::DiagnosticsEngine &diags,
                                           clang::SourceManager &srcMgr,
                                           clang::FileManager &fileMgr)
    : mlir::ScopedDiagnosticHandler(ctx), Diags(diags), SrcMgr(srcMgr),
      FileMgr(fileMgr) {
  setHandler([this](mlir::Diagnostic &D) { return handle(D); });
}

clang::SourceLocation CIRDiagnosticHandler::translateLoc(mlir::Location loc) {
  // Walk common location wrappers to reach a usable file/line/column triple.
  // Anything we can't translate becomes an invalid SourceLocation, which the
  // diagnostics engine renders without a source line.
  if (auto file = mlir::dyn_cast<mlir::FileLineColLoc>(loc)) {
    // SourceManager::translateFileLineCol requires 1-based line/column.
    // Module-level locations carry (0, 0); fall through unattached.
    if (file.getLine() == 0 || file.getColumn() == 0)
      return clang::SourceLocation();
    auto fileRef = FileMgr.getOptionalFileRef(file.getFilename().getValue());
    if (!fileRef)
      return clang::SourceLocation();
    return SrcMgr.translateFileLineCol(&fileRef->getFileEntry(), file.getLine(),
                                       file.getColumn());
  }
  if (auto fused = mlir::dyn_cast<mlir::FusedLoc>(loc)) {
    for (mlir::Location child : fused.getLocations()) {
      clang::SourceLocation translated = translateLoc(child);
      if (translated.isValid())
        return translated;
    }
    return clang::SourceLocation();
  }
  if (auto callsite = mlir::dyn_cast<mlir::CallSiteLoc>(loc))
    return translateLoc(callsite.getCallee());
  if (auto named = mlir::dyn_cast<mlir::NameLoc>(loc))
    return translateLoc(named.getChildLoc());
  // OpaqueLoc / UnknownLoc and anything else fall through unattached.
  return clang::SourceLocation();
}

void CIRDiagnosticHandler::emit(mlir::Diagnostic &diag, bool isNote) {
  unsigned diagID;
  if (isNote) {
    diagID = clang::diag::note_cir_mlir_diagnostic;
  } else {
    switch (diag.getSeverity()) {
    case mlir::DiagnosticSeverity::Error:
      diagID = clang::diag::err_cir_mlir_diagnostic;
      break;
    case mlir::DiagnosticSeverity::Warning:
      diagID = clang::diag::warn_cir_mlir_diagnostic;
      break;
    case mlir::DiagnosticSeverity::Remark:
      diagID = clang::diag::remark_cir_mlir_diagnostic;
      break;
    case mlir::DiagnosticSeverity::Note:
      diagID = clang::diag::note_cir_mlir_diagnostic;
      break;
    }
  }
  Diags.Report(translateLoc(diag.getLocation()), diagID) << diag.str();
}

mlir::LogicalResult CIRDiagnosticHandler::handle(mlir::Diagnostic &diag) {
  emit(diag, /*isNote=*/false);
  for (mlir::Diagnostic &note : diag.getNotes())
    emit(note, /*isNote=*/true);
  return mlir::success();
}

} // namespace cir
