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

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

namespace clang {
class ASTContext;
class CodeGenOptions;
class Decl;
class DiagnosticsEngine;
class LangOptions;
class TargetInfo;
} // namespace clang

using namespace clang;
namespace cir {

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
  /// Hold Clang AST information.
  clang::ASTContext &astCtx;

  const clang::LangOptions &langOpts;

  /// A "module" matches a c/cpp source file: containing a list of functions.
  mlir::ModuleOp theModule;

  const clang::TargetInfo &target;

public:
  void buildTopLevelDecl(clang::Decl *decl);
};
} // namespace cir

#endif // LLVM_CLANG_LIB_CIR_CODEGEN_CIRGENMODULE_H
