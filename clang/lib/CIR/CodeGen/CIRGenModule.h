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

#include "CIRGenBuilder.h"
#include "CIRGenTypeCache.h"
#include "CIRGenTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/ADT/StringRef.h"

namespace clang {
class ASTContext;
class CodeGenOptions;
class Decl;
class GlobalDecl;
class LangOptions;
class TargetInfo;
class VarDecl;

namespace CIRGen {

/// This class organizes the cross-function state that is used while generating
/// CIR code.
class CIRGenModule : public CIRGenTypeCache {
  CIRGenModule(CIRGenModule &) = delete;
  CIRGenModule &operator=(CIRGenModule &) = delete;

public:
  CIRGenModule(mlir::MLIRContext &mlirContext, clang::ASTContext &astContext,
               const clang::CodeGenOptions &cgo,
               clang::DiagnosticsEngine &diags);

  ~CIRGenModule() = default;

private:
  CIRGenBuilderTy builder;

  /// Hold Clang AST information.
  clang::ASTContext &astContext;

  const clang::LangOptions &langOpts;

  /// A "module" matches a c/cpp source file: containing a list of functions.
  mlir::ModuleOp theModule;

  clang::DiagnosticsEngine &diags;

  const clang::TargetInfo &target;

  CIRGenTypes genTypes;

public:
  mlir::ModuleOp getModule() const { return theModule; }
  CIRGenBuilderTy &getBuilder() { return builder; }
  clang::ASTContext &getASTContext() const { return astContext; }
  CIRGenTypes &getTypes() { return genTypes; }
  mlir::MLIRContext &getMLIRContext() { return *builder.getContext(); }

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
  void emitGlobalVarDefinition(const clang::VarDecl *vd,
                               bool isTentative = false);

  /// Helpers to emit "not yet implemented" error diagnostics
  DiagnosticBuilder errorNYI(SourceLocation, llvm::StringRef);

  template <typename T>
  DiagnosticBuilder errorNYI(SourceLocation loc, llvm::StringRef feature,
                             const T &name) {
    unsigned diagID =
        diags.getCustomDiagID(DiagnosticsEngine::Error,
                              "ClangIR code gen Not Yet Implemented: %0: %1");
    return diags.Report(loc, diagID) << feature << name;
  }

  DiagnosticBuilder errorNYI(SourceRange, llvm::StringRef);

  template <typename T>
  DiagnosticBuilder errorNYI(SourceRange loc, llvm::StringRef feature,
                             const T &name) {
    return errorNYI(loc.getBegin(), feature, name) << loc;
  }
};
} // namespace CIRGen

} // namespace clang

#endif // LLVM_CLANG_LIB_CIR_CODEGEN_CIRGENMODULE_H
