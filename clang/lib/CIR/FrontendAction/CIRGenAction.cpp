//===--- CIRGenAction.cpp - LLVM Code generation Frontend Action ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/CIRGenerator.h"
#include "clang/CIRFrontendAction/CIRGenAction.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

using namespace cir;
using namespace clang;

namespace cir {

class CIRGenConsumer : public clang::ASTConsumer {

  virtual void anchor();

  std::unique_ptr<CIRGenerator> gen;
  bool HandleTopLevelDecl(DeclGroupRef D) override {
    gen->HandleTopLevelDecl(D);
    return true;
  }
};
} // namespace cir

void CIRGenConsumer::anchor() {}

CIRGenAction::CIRGenAction(OutputType act, mlir::MLIRContext *mlirContext)
    : mlirContext(mlirContext ? mlirContext : new mlir::MLIRContext),
      action(act) {}

CIRGenAction::~CIRGenAction() { mlirModule.reset(); }

std::unique_ptr<ASTConsumer>
CIRGenAction::CreateASTConsumer(CompilerInstance &ci, StringRef inputFile) {}

void EmitCIRAction::anchor() {}
EmitCIRAction::EmitCIRAction(mlir::MLIRContext *_MLIRContext)
    : CIRGenAction(OutputType::EmitCIR, _MLIRContext) {}
