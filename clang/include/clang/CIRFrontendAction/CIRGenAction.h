//===---- CIRGenAction.h - CIR Code Generation Frontend Action -*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CIR_CIRGENACTION_H
#define LLVM_CLANG_CIR_CIRGENACTION_H

#include "clang/Frontend/FrontendAction.h"

namespace mlir {
class MLIRContext;
class ModuleOp;
template <typename T> class OwningOpRef;
} // namespace mlir

namespace cir {

class CIRGenAction : public clang::ASTFrontendAction {
public:
  enum class OutputType {
    EmitCIR,
  };

private:
  friend class CIRGenConsumer;

  // TODO: this is redundant but just using the OwningModuleRef requires more of
  // clang against MLIR. Hide this somewhere else.
  std::unique_ptr<mlir::OwningOpRef<mlir::ModuleOp>> mlirModule;

  mlir::MLIRContext *mlirContext;

protected:
  CIRGenAction(OutputType action, mlir::MLIRContext *mlirContext = nullptr);

  void foo() {

  }

  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &CI,
                    llvm::StringRef InFile) override;

public:
  ~CIRGenAction() override;
  OutputType action;
};

class EmitCIRAction : public CIRGenAction {
  virtual void anchor();

public:
  EmitCIRAction(mlir::MLIRContext *mlirCtx = nullptr);
};

} // namespace cir

#endif
