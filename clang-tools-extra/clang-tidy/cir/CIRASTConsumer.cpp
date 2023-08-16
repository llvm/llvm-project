//===--- clang-tidy/cir-tidy/CIRASTConsumer.cpp ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CIRASTConsumer.h"

#include "../utils/OptionsUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "clang/CIR/Dialect/Passes.h"
#include <algorithm>

using namespace clang;
using namespace clang::tidy;

namespace clang::tidy::cir {

/// CIR AST Consumer
CIRASTConsumer::CIRASTConsumer(CompilerInstance &CI, StringRef inputFile,
                               clang::tidy::ClangTidyContext &Context)
    : Context(Context),
      OptsView(ClangTidyCheck::OptionsView(
          LifetimeCheckName, Context.getOptions().CheckOptions, &Context)) {
  // Setup CIR codegen options via config specified information.
  CI.getCodeGenOpts().ClangIRBuildDeferredThreshold =
      OptsView.get("CodeGenBuildDeferredThreshold", 500U);
  CI.getCodeGenOpts().ClangIRSkipFunctionsFromSystemHeaders =
      OptsView.get("CodeGenSkipFunctionsFromSystemHeaders", false);

  Gen = std::make_unique<::cir::CIRGenerator>(CI.getDiagnostics(), nullptr,
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

    clang::SourceLocation getClangFromFileLineCol(mlir::FileLineColLoc loc) {
      clang::SourceLocation clangLoc;
      FileManager &fileMgr = clangSrcMgr.getFileManager();
      assert(loc && "not a valid mlir::FileLineColLoc");
      // The column and line may be zero to represent unknown column and/or
      // unknown line/column information.
      if (loc.getLine() == 0 || loc.getColumn() == 0) {
        llvm_unreachable("How should we workaround this?");
        return clangLoc;
      }
      if (auto FE = fileMgr.getFile(loc.getFilename())) {
        return clangSrcMgr.translateFileLineCol(*FE, loc.getLine(),
                                                loc.getColumn());
      }
      llvm_unreachable("location doesn't map to a file?");
    }

    clang::SourceLocation getClangSrcLoc(mlir::Location loc) {
      // Direct maps into a clang::SourceLocation.
      if (auto fileLoc = loc.dyn_cast<mlir::FileLineColLoc>()) {
        return getClangFromFileLineCol(fileLoc);
      }

      // FusedLoc needs to be decomposed but the canonical one
      // is the first location, we handle source ranges somewhere
      // else.
      if (auto fileLoc = loc.dyn_cast<mlir::FusedLoc>()) {
        auto locArray = fileLoc.getLocations();
        assert(locArray.size() > 0 && "expected multiple locs");
        return getClangFromFileLineCol(
            locArray[0].dyn_cast<mlir::FileLineColLoc>());
      }

      // Many loc styles are yet to be handled.
      if (auto fileLoc = loc.dyn_cast<mlir::UnknownLoc>()) {
        llvm_unreachable("mlir::UnknownLoc not implemented!");
      }
      if (auto fileLoc = loc.dyn_cast<mlir::CallSiteLoc>()) {
        llvm_unreachable("mlir::CallSiteLoc not implemented!");
      }
      llvm_unreachable("Unknown location style");
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
      auto clangBeginLoc = getClangSrcLoc(diag.getLocation());
      tidyCtx.diag(LifetimeCheckName, clangBeginLoc, diag.str(),
                   translateToClangDiagLevel(diag.getSeverity()));
      for (const auto &note : diag.getNotes()) {
        auto clangNoteBeginLoc = getClangSrcLoc(note.getLocation());
        tidyCtx.diag(LifetimeCheckName, clangNoteBeginLoc, note.str(),
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
      setHandler(
          [this](mlir::Diagnostic &diag) { emitClangTidyDiagnostic(diag); });
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

  if (Context.isCheckEnabled(LifetimeCheckName))
    pm.addPass(mlir::createLifetimeCheckPass(remarks, hist, hLimit, &C));

  bool Result = !mlir::failed(pm.run(mlirMod));
  if (!Result)
    llvm::report_fatal_error(
        "The pass manager failed to run pass on the module!");
}
} // namespace clang::tidy::cir
