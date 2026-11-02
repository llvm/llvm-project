//===---- CIRGenAction.h - CIR Code Generation Frontend Action -*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CIR_CIRGENACTION_H
#define LLVM_CLANG_CIR_CIRGENACTION_H

#include "clang/CodeGen/ModuleLinker.h"
#include "clang/Frontend/FrontendAction.h"
#include "llvm/ADT/SmallVector.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"

namespace llvm {
class LLVMContext;
} // namespace llvm

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
    EmitCIRBC,
    EmitLLVM,
    EmitBC,
    EmitObj,
  };

private:
  friend class CIRGenConsumer;

  mlir::OwningOpRef<mlir::ModuleOp> MLIRMod;

  std::unique_ptr<llvm::LLVMContext> Ctx;
  llvm::SmallVector<clang::LinkModule> LinkModules;

protected:
  CIRGenAction(OutputType Action);

  bool BeginSourceFileAction(clang::CompilerInstance &CI) override;

  bool hasCIRSupport() const override { return true; }

  void ExecuteAction() override;

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
  EmitCIRAction();
};

class EmitCIRBCAction : public CIRGenAction {
  virtual void anchor();

public:
  EmitCIRBCAction();
};

class EmitLLVMAction : public CIRGenAction {
  virtual void anchor();

public:
  EmitLLVMAction();
};

class EmitBCAction : public CIRGenAction {
  virtual void anchor();

public:
  EmitBCAction();
};

class EmitAssemblyAction : public CIRGenAction {
  virtual void anchor();

public:
  EmitAssemblyAction();
};

class EmitObjAction : public CIRGenAction {
  virtual void anchor();

public:
  EmitObjAction();
};

} // namespace cir

#endif
