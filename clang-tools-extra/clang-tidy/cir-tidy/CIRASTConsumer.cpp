//===--- clang-tidy/cir-tidy/CIRASTConsumer.cpp ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CIRASTConsumer.h"
#include "CIRChecks.h"

#include "../utils/OptionsUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "clang/CIR/Dialect/Passes.h"
#include <algorithm>

using namespace clang;
using namespace clang::tidy;

namespace cir {
namespace tidy {

CIRASTConsumer::CIRASTConsumer(CompilerInstance &CI, StringRef inputFile,
                               clang::tidy::ClangTidyContext &Context)
    : Context(Context),
      OptsView(ClangTidyCheck::OptionsView(cir::checks::LifetimeCheckName,
                                           Context.getOptions().CheckOptions,
                                           &Context)) {
  // Setup CIR codegen options via config specified information.
  CI.getCodeGenOpts().ClangIRBuildDeferredThreshold =
      OptsView.get("CodeGenBuildDeferredThreshold", 500U);
  CI.getCodeGenOpts().ClangIRSkipFunctionsFromSystemHeaders =
      OptsView.get("CodeGenSkipFunctionsFromSystemHeaders", false);

  Gen = std::make_unique<CIRGenerator>(CI.getDiagnostics(), nullptr,
                                       CI.getCodeGenOpts());
}

bool CIRASTConsumer::HandleTopLevelDecl(DeclGroupRef D) {
  PrettyStackTraceDecl CrashInfo(*D.begin(), SourceLocation(),
                                 AstContext->getSourceManager(),
                                 "CIR generation of declaration");
  Gen->HandleTopLevelDecl(D);
  return true;
}

void CIRASTConsumer::Initialize(ASTContext &Context) {
  AstContext = &Context;
  Gen->Initialize(Context);
}

void CIRASTConsumer::HandleTranslationUnit(ASTContext &C) {
  Gen->HandleTranslationUnit(C);
  Gen->verifyModule();

  mlir::ModuleOp mlirMod = Gen->getModule();
  std::unique_ptr<mlir::MLIRContext> mlirCtx = Gen->takeContext();

  mlir::OpPrintingFlags flags;
  flags.enableDebugInfo(/*prettyForm=*/false);

  clang::SourceManager &clangSrcMgr = C.getSourceManager();
  FileID MainFileID = clangSrcMgr.getMainFileID();

  llvm::MemoryBufferRef MainFileBuf = clangSrcMgr.getBufferOrFake(MainFileID);
  std::unique_ptr<llvm::MemoryBuffer> FileBuf =
      llvm::MemoryBuffer::getMemBuffer(MainFileBuf);

  llvm::SourceMgr llvmSrcMgr;
  llvmSrcMgr.AddNewSourceBuffer(std::move(FileBuf), llvm::SMLoc());

  class CIRTidyDiagnosticHandler : public mlir::SourceMgrDiagnosticHandler {
    clang::tidy::ClangTidyContext &tidyCtx;
    clang::SourceManager &clangSrcMgr;

    clang::SourceLocation getClangSrcLoc(mlir::Location loc) {
      clang::SourceLocation clangLoc;
      FileManager &fileMgr = clangSrcMgr.getFileManager();

      auto fileLoc = loc.dyn_cast<mlir::FileLineColLoc>();
      if (!fileLoc)
        return clangLoc;
      // The column and line may be zero to represent unknown column and/or
      // unknown line/column information.
      if (fileLoc.getLine() == 0 || fileLoc.getColumn() == 0)
        return clangLoc;
      if (auto FE = fileMgr.getFile(fileLoc.getFilename())) {
        return clangSrcMgr.translateFileLineCol(*FE, fileLoc.getLine(),
                                                fileLoc.getColumn());
      }
      return clangLoc;
    }

    clang::DiagnosticIDs::Level
    translateToClangDiagLevel(const mlir::DiagnosticSeverity &sev) {
      switch (sev) {
      case mlir::DiagnosticSeverity::Note:
        return clang::DiagnosticIDs::Level::Note;
      case mlir::DiagnosticSeverity::Warning:
        return clang::DiagnosticIDs::Level::Warning;
      case mlir::DiagnosticSeverity::Error:
        return clang::DiagnosticIDs::Level::Error;
      case mlir::DiagnosticSeverity::Remark:
        return clang::DiagnosticIDs::Level::Remark;
      }
      llvm_unreachable("should not get here!");
    }

  public:
    void emitClangTidyDiagnostic(mlir::Diagnostic &diag) {
      tidyCtx.diag(cir::checks::LifetimeCheckName,
                   getClangSrcLoc(diag.getLocation()), diag.str(),
                   translateToClangDiagLevel(diag.getSeverity()));
      for (const auto &note : diag.getNotes()) {
        tidyCtx.diag(cir::checks::LifetimeCheckName,
                     getClangSrcLoc(note.getLocation()), note.str(),
                     translateToClangDiagLevel(note.getSeverity()));
      }
    }

    CIRTidyDiagnosticHandler(llvm::SourceMgr &mgr, mlir::MLIRContext *ctx,
                             clang::tidy::ClangTidyContext &tidyContext,
                             clang::SourceManager &clangMgr,
                             ShouldShowLocFn &&shouldShowLocFn = {})
        : SourceMgrDiagnosticHandler(mgr, ctx, llvm::errs(),
                                     std::move(shouldShowLocFn)),
          tidyCtx(tidyContext), clangSrcMgr(clangMgr) {
      setHandler([this](mlir::Diagnostic &diag) {
        // Emit diagnostic to llvm::errs() but also populate Clang
        emitClangTidyDiagnostic(diag);
        emitDiagnostic(diag);
      });
    }
    ~CIRTidyDiagnosticHandler() = default;
  };

  // Use a custom diagnostic handler that can allow both regular printing to
  // stderr but also populates clang-tidy context with diagnostics (and allow
  // for instance, diagnostics to be later converted to YAML).
  CIRTidyDiagnosticHandler sourceMgrHandler(llvmSrcMgr, mlirCtx.get(), Context,
                                            clangSrcMgr);

  mlir::PassManager pm(mlirCtx.get());
  pm.addPass(mlir::createMergeCleanupsPass());

  auto remarks =
      utils::options::parseStringList(OptsView.get("RemarksList", ""));
  auto hist =
      utils::options::parseStringList(OptsView.get("HistoryList", "all"));
  auto hLimit = OptsView.get("HistLimit", 1U);

  if (Context.isCheckEnabled(cir::checks::LifetimeCheckName))
    pm.addPass(mlir::createLifetimeCheckPass(remarks, hist, hLimit, &C));

  bool Result = !mlir::failed(pm.run(mlirMod));
  if (!Result)
    llvm::report_fatal_error(
        "The pass manager failed to run pass on the module!");
}
} // namespace tidy
} // namespace cir
