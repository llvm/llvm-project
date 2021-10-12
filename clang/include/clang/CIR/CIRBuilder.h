//===- CIRBuilder.h - CIR Generation from Clang AST -----------------------===//
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

#ifndef CLANG_CIRBUILDER_H_
#define CLANG_CIRBUILDER_H_

#include "clang/AST/ASTConsumer.h"
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
class CIRBuildImpl;
class CIRGenTypes;

class CIRContext : public clang::ASTConsumer {
public:
  CIRContext();
  ~CIRContext();
  void Initialize(clang::ASTContext &Context) override;
  bool EmitFunction(const clang::FunctionDecl *FD);

  bool HandleTopLevelDecl(clang::DeclGroupRef D) override;
  void HandleTranslationUnit(clang::ASTContext &Ctx) override;

  mlir::ModuleOp getModule();
  std::unique_ptr<mlir::MLIRContext> takeContext() {
    return std::move(mlirCtx);
  };

  void verifyModule();

private:
  std::unique_ptr<mlir::MLIRContext> mlirCtx;
  std::unique_ptr<CIRBuildImpl> builder;

  clang::ASTContext *astCtx;
};

} // namespace cir

#endif // CLANG_CIRBUILDER_H_
