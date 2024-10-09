//===--- CIRGenAction.cpp - LLVM Code generation Frontend Action ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/FrontendAction/CIRGenAction.h"
#include "clang/CIR/CIRGenerator.h"
#include "clang/Frontend/CompilerInstance.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

using namespace cir;
using namespace clang;

namespace cir {

class CIRGenConsumer : public clang::ASTConsumer {

  virtual void anchor();

  std::unique_ptr<raw_pwrite_stream> OutputStream;

  IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS;
  std::unique_ptr<CIRGenerator> Gen;

public:
  CIRGenConsumer(CIRGenAction::OutputType Action,
                 DiagnosticsEngine &DiagnosticsEngine,
                 IntrusiveRefCntPtr<llvm::vfs::FileSystem> VFS,
                 const HeaderSearchOptions &HeaderSearchOptions,
                 const CodeGenOptions &CodeGenOptions,
                 const TargetOptions &TargetOptions,
                 const LangOptions &LangOptions,
                 const FrontendOptions &FEOptions,
                 std::unique_ptr<raw_pwrite_stream> OS)
      : OutputStream(std::move(OS)), FS(VFS),
        Gen(std::make_unique<CIRGenerator>(DiagnosticsEngine, std::move(VFS),
                                           CodeGenOptions)) {}

  bool HandleTopLevelDecl(DeclGroupRef D) override {
    Gen->HandleTopLevelDecl(D);
    return true;
  }
};
} // namespace cir

void CIRGenConsumer::anchor() {}

CIRGenAction::CIRGenAction(OutputType Act, mlir::MLIRContext *MLIRCtx)
    : MLIRCtx(MLIRCtx ? MLIRCtx : new mlir::MLIRContext), Action(Act) {}

CIRGenAction::~CIRGenAction() { MLIRMod.release(); }

std::unique_ptr<ASTConsumer>
CIRGenAction::CreateASTConsumer(CompilerInstance &CI, StringRef InFile) {
  std::unique_ptr<llvm::raw_pwrite_stream> Out = CI.takeOutputStream();

  auto Result = std::make_unique<cir::CIRGenConsumer>(
      Action, CI.getDiagnostics(), &CI.getVirtualFileSystem(),
      CI.getHeaderSearchOpts(), CI.getCodeGenOpts(), CI.getTargetOpts(),
      CI.getLangOpts(), CI.getFrontendOpts(), std::move(Out));

  return Result;
}

void EmitCIRAction::anchor() {}
EmitCIRAction::EmitCIRAction(mlir::MLIRContext *MLIRCtx)
    : CIRGenAction(OutputType::EmitCIR, MLIRCtx) {}
