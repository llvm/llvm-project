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
#include <memory>

namespace llvm {
class LLVMIRContext;
}

namespace mlir {
class MLIRContext;
class ModuleOp;
} // namespace mlir

namespace cir {
class CIRGenConsumer;
class CIRGenerator;

class CIRGenAction : public clang::ASTFrontendAction {
public:
  enum class OutputType { EmitAssembly, EmitCIR, EmitLLVM, None };

private:
  friend class CIRGenConsumer;

  std::unique_ptr<mlir::ModuleOp> TheModule;

  mlir::MLIRContext *MLIRContext;
  bool OwnsVMContext;

  std::unique_ptr<mlir::ModuleOp> loadModule(llvm::MemoryBufferRef MBRef);

protected:
  CIRGenAction(OutputType action, mlir::MLIRContext *_MLIRContext = nullptr);

  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &CI,
                    llvm::StringRef InFile) override;

  void ExecuteAction() override;

  void EndSourceFileAction() override;

public:
  ~CIRGenAction() override;

  CIRGenConsumer *cgConsumer;
  OutputType action;
};

class EmitLLVMAction : public CIRGenAction {
  virtual void anchor();

public:
  EmitLLVMAction(mlir::MLIRContext *mlirCtx = nullptr);
};

class EmitCIRAction : public CIRGenAction {
  virtual void anchor();

public:
  EmitCIRAction(mlir::MLIRContext *mlirCtx = nullptr);
};

class EmitCIROnlyAction : public CIRGenAction {
  virtual void anchor();

public:
  EmitCIROnlyAction(mlir::MLIRContext *mlirCtx = nullptr);
};

class EmitAssemblyAction : public CIRGenAction {
  virtual void anchor();

public:
  EmitAssemblyAction(mlir::MLIRContext *mlirCtx = nullptr);
};

} // namespace cir

#endif
