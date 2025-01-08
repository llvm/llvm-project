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

  CIRGenAction::OutputType Action;

  std::unique_ptr<raw_pwrite_stream> OutputStream;

  ASTContext *Context{nullptr};
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
      : Action(Action), OutputStream(std::move(OS)), FS(VFS),
        Gen(std::make_unique<CIRGenerator>(DiagnosticsEngine, std::move(VFS),
                                           CodeGenOptions)) {}

  void Initialize(ASTContext &Ctx) override {
    assert(!Context && "initialized multiple times");
    Context = &Ctx;
    Gen->Initialize(Ctx);
  }

  bool HandleTopLevelDecl(DeclGroupRef D) override {
    Gen->HandleTopLevelDecl(D);
    return true;
  }

  void HandleTranslationUnit(ASTContext &C) override {
    Gen->HandleTranslationUnit(C);
    mlir::ModuleOp MlirModule = Gen->getModule();
    switch (Action) {
    case CIRGenAction::OutputType::EmitCIR:
      if (OutputStream && MlirModule) {
        mlir::OpPrintingFlags Flags;
        Flags.enableDebugInfo(/*enable=*/true, /*prettyForm=*/false);
        MlirModule->print(*OutputStream, Flags);
      }
      break;
    }
  }
};
} // namespace cir

void CIRGenConsumer::anchor() {}

CIRGenAction::CIRGenAction(OutputType Act, mlir::MLIRContext *MLIRCtx)
    : MLIRCtx(MLIRCtx ? MLIRCtx : new mlir::MLIRContext), Action(Act) {}

CIRGenAction::~CIRGenAction() { MLIRMod.release(); }

static std::unique_ptr<raw_pwrite_stream>
getOutputStream(CompilerInstance &CI, StringRef InFile,
                CIRGenAction::OutputType Action) {
  switch (Action) {
  case CIRGenAction::OutputType::EmitCIR:
    return CI.createDefaultOutputFile(false, InFile, "cir");
  }
  llvm_unreachable("Invalid CIRGenAction::OutputType");
}

std::unique_ptr<ASTConsumer>
CIRGenAction::CreateASTConsumer(CompilerInstance &CI, StringRef InFile) {
  std::unique_ptr<llvm::raw_pwrite_stream> Out = CI.takeOutputStream();

  if (!Out)
    Out = getOutputStream(CI, InFile, Action);

  auto Result = std::make_unique<cir::CIRGenConsumer>(
      Action, CI.getDiagnostics(), &CI.getVirtualFileSystem(),
      CI.getHeaderSearchOpts(), CI.getCodeGenOpts(), CI.getTargetOpts(),
      CI.getLangOpts(), CI.getFrontendOpts(), std::move(Out));

  return Result;
}

void EmitCIRAction::anchor() {}
EmitCIRAction::EmitCIRAction(mlir::MLIRContext *MLIRCtx)
    : CIRGenAction(OutputType::EmitCIR, MLIRCtx) {}
