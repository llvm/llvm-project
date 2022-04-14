//===- CIRGenerator.h - CIR Generation from Clang AST ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares a simple interface to perform CIR generation from Clang
// AST
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIRGENERATOR_H_
#define CLANG_CIRGENERATOR_H_

#include "clang/AST/ASTConsumer.h"
#include "clang/Basic/CodeGenOptions.h"

#include "llvm/Support/ToolOutputFile.h"

#include <memory>

namespace mlir {
class MLIRContext;
class ModuleOp;
class OwningModuleRef;
} // namespace mlir

namespace clang {
class ASTContext;
class DeclGroupRef;
class FunctionDecl;
} // namespace clang

namespace cir {
class CIRGenModule;
class CIRGenTypes;

class CIRGenerator : public clang::ASTConsumer {
  virtual void anchor();
  clang::DiagnosticsEngine &Diags;
  clang::ASTContext *astCtx;

  const clang::CodeGenOptions codeGenOpts; // Intentionally copied in.

protected:
  std::unique_ptr<mlir::MLIRContext> mlirCtx;
  std::unique_ptr<CIRGenModule> CGM;

private:
public:
  CIRGenerator(clang::DiagnosticsEngine &diags,
               const clang::CodeGenOptions &CGO);
  ~CIRGenerator();
  void Initialize(clang::ASTContext &Context) override;
  bool EmitFunction(const clang::FunctionDecl *FD);

  bool HandleTopLevelDecl(clang::DeclGroupRef D) override;
  void HandleTranslationUnit(clang::ASTContext &Ctx) override;
  void HandleInlineFunctionDefinition(clang::FunctionDecl *D) override;
  void HandleTagDeclDefinition(clang::TagDecl *D) override;
  void HandleTagDeclRequiredDefinition(const clang::TagDecl *D) override;

  mlir::ModuleOp getModule();
  std::unique_ptr<mlir::MLIRContext> takeContext() {
    return std::move(mlirCtx);
  };

  void verifyModule();

};

} // namespace cir

#endif // CLANG_CIRGENERATOR_H_
