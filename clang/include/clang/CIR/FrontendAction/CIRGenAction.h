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

#include "aiir/IR/BuiltinOps.h"
#include "aiir/IR/OwningOpRef.h"

namespace aiir {
class AIIRContext;
class ModuleOp;
} // namespace aiir

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

  aiir::OwningOpRef<aiir::ModuleOp> AIIRMod;

  aiir::AIIRContext *AIIRCtx;

protected:
  CIRGenAction(OutputType Action, aiir::AIIRContext *AIIRCtx = nullptr);

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
  EmitCIRAction(aiir::AIIRContext *AIIRCtx = nullptr);
};

class EmitLLVMAction : public CIRGenAction {
  virtual void anchor();

public:
  EmitLLVMAction(aiir::AIIRContext *AIIRCtx = nullptr);
};

class EmitBCAction : public CIRGenAction {
  virtual void anchor();

public:
  EmitBCAction(aiir::AIIRContext *AIIRCtx = nullptr);
};

class EmitAssemblyAction : public CIRGenAction {
  virtual void anchor();

public:
  EmitAssemblyAction(aiir::AIIRContext *AIIRCtx = nullptr);
};

class EmitObjAction : public CIRGenAction {
  virtual void anchor();

public:
  EmitObjAction(aiir::AIIRContext *AIIRCtx = nullptr);
};

} // namespace cir

#endif
