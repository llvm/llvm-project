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

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"

namespace mlir {
class MLIRContext;
class ModuleOp;
} // namespace mlir

namespace cir {
class CIRGenConsumer;

class CIRGenAction : public clang::ASTFrontendAction {
public:
  enum class OutputType {
    EmitAssembly,
    EmitCIR,
    EmitLLVM,
    EmitBC,
    EmitObj,
  };

private:
  friend class CIRGenConsumer;

  mlir::OwningOpRef<mlir::ModuleOp> MLIRMod;

  mlir::MLIRContext *MLIRCtx;

protected:
  CIRGenAction(OutputType Action, mlir::MLIRContext *MLIRCtx = nullptr);

  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &CI,
                    llvm::StringRef InFile) override;

public:
  ~CIRGenAction() override;

  OutputType Action;
};

class EmitCIRAction : public CIRGenAction {
  virtual void anchor();

public:
  EmitCIRAction(mlir::MLIRContext *MLIRCtx = nullptr);
};

class EmitLLVMAction : public CIRGenAction {
  virtual void anchor();

public:
  EmitLLVMAction(mlir::MLIRContext *MLIRCtx = nullptr);
};

class EmitBCAction : public CIRGenAction {
  virtual void anchor();

public:
  EmitBCAction(mlir::MLIRContext *MLIRCtx = nullptr);
};

class EmitAssemblyAction : public CIRGenAction {
  virtual void anchor();

public:
  EmitAssemblyAction(mlir::MLIRContext *MLIRCtx = nullptr);
};

class EmitObjAction : public CIRGenAction {
  virtual void anchor();

public:
  EmitObjAction(mlir::MLIRContext *MLIRCtx = nullptr);
};

} // namespace cir

#endif
