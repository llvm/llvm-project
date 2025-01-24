//===--- CIRGenModule.h - Per-Module state for CIR gen ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the internal per-translation-unit state used for CIR translation.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CIR_CODEGEN_CIRGENMODULE_H
#define LLVM_CLANG_LIB_CIR_CODEGEN_CIRGENMODULE_H

#include "CIRGenTypeCache.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/StringRef.h"

namespace clang {
class ASTContext;
class CodeGenOptions;
class Decl;
class DiagnosticBuilder;
class DiagnosticsEngine;
class GlobalDecl;
class LangOptions;
class SourceLocation;
class SourceRange;
class TargetInfo;

namespace CIRGen {

/// This class organizes the cross-function state that is used while generating
/// CIR code.
class CIRGenModule : public CIRGenTypeCache {
  CIRGenModule(CIRGenModule &) = delete;
  CIRGenModule &operator=(CIRGenModule &) = delete;

public:
  CIRGenModule(mlir::MLIRContext &context, clang::ASTContext &astctx,
               const clang::CodeGenOptions &cgo,
               clang::DiagnosticsEngine &diags);

  ~CIRGenModule() = default;

private:
  // TODO(CIR) 'builder' will change to CIRGenBuilderTy once that type is
  // defined
  mlir::OpBuilder builder;

  /// Hold Clang AST information.
  clang::ASTContext &astCtx;

  const clang::LangOptions &langOpts;

  /// A "module" matches a c/cpp source file: containing a list of functions.
  mlir::ModuleOp theModule;

  clang::DiagnosticsEngine &diags;

  const clang::TargetInfo &target;

public:
  mlir::ModuleOp getModule() const { return theModule; }

  /// Helpers to convert the presumed location of Clang's SourceLocation to an
  /// MLIR Location.
  mlir::Location getLoc(clang::SourceLocation cLoc);
  mlir::Location getLoc(clang::SourceRange cRange);

  void emitTopLevelDecl(clang::Decl *decl);

  /// Emit code for a single global function or variable declaration. Forward
  /// declarations are emitted lazily.
  void emitGlobal(clang::GlobalDecl gd);

  void emitGlobalDefinition(clang::GlobalDecl gd,
                            mlir::Operation *op = nullptr);
  void emitGlobalFunctionDefinition(clang::GlobalDecl gd, mlir::Operation *op);

  /// Helpers to emit "not yet implemented" error diagnostics
  DiagnosticBuilder errorNYI(llvm::StringRef);
  DiagnosticBuilder errorNYI(SourceLocation, llvm::StringRef);
  DiagnosticBuilder errorNYI(SourceLocation, llvm::StringRef, llvm::StringRef);
  DiagnosticBuilder errorNYI(SourceRange, llvm::StringRef);
  DiagnosticBuilder errorNYI(SourceRange, llvm::StringRef, llvm::StringRef);
};
} // namespace CIRGen

} // namespace clang

#endif // LLVM_CLANG_LIB_CIR_CODEGEN_CIRGENMODULE_H
