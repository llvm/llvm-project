//===--- CIRGenAction.cpp - LLVM Code generation Frontend Action ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CIRFrontendAction/CIRGenAction.h"
#include "clang/CIR/CIRGenerator.h"
#include "clang/Frontend/CompilerInstance.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

using namespace cir;
using namespace clang;

namespace cir {

class CIRGenConsumer : public clang::ASTConsumer {

  virtual void anchor();

  CIRGenAction::OutputType action;

  DiagnosticsEngine &diagnosticsEngine;
  const HeaderSearchOptions &headerSearchOptions;
  const CodeGenOptions &codeGenOptions;
  const TargetOptions &targetOptions;
  const LangOptions &langOptions;
  const FrontendOptions &feOptions;

  std::unique_ptr<raw_pwrite_stream> outputStream;

  ASTContext *astContext{nullptr};
  IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS;
  std::unique_ptr<CIRGenerator> gen;

public:
  CIRGenConsumer(CIRGenAction::OutputType action,
                 DiagnosticsEngine &diagnosticsEngine,
                 IntrusiveRefCntPtr<llvm::vfs::FileSystem> VFS,
                 const HeaderSearchOptions &headerSearchOptions,
                 const CodeGenOptions &codeGenOptions,
                 const TargetOptions &targetOptions,
                 const LangOptions &langOptions,
                 const FrontendOptions &feOptions,
                 std::unique_ptr<raw_pwrite_stream> os)
      : action(action), diagnosticsEngine(diagnosticsEngine),
        headerSearchOptions(headerSearchOptions),
        codeGenOptions(codeGenOptions), targetOptions(targetOptions),
        langOptions(langOptions), feOptions(feOptions),
        outputStream(std::move(os)), FS(VFS),
        gen(std::make_unique<CIRGenerator>(diagnosticsEngine, std::move(VFS),
                                           codeGenOptions)) {}

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
CIRGenAction::CreateASTConsumer(CompilerInstance &ci, StringRef inputFile) {
  auto out = ci.takeOutputStream();

  auto Result = std::make_unique<cir::CIRGenConsumer>(
      action, ci.getDiagnostics(), &ci.getVirtualFileSystem(),
      ci.getHeaderSearchOpts(), ci.getCodeGenOpts(), ci.getTargetOpts(),
      ci.getLangOpts(), ci.getFrontendOpts(), std::move(out));
  cgConsumer = Result.get();

  return std::move(Result);
}

void EmitCIRAction::anchor() {}
EmitCIRAction::EmitCIRAction(mlir::MLIRContext *_MLIRContext)
    : CIRGenAction(OutputType::EmitCIR, _MLIRContext) {}
