//===--- Lifetime.cpp - clang-tidy ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Lifetime.h"
#include "../utils/OptionsUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclGroup.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/CIR/CIRGenerator.h"
#include "clang/CIR/Dialect/Passes.h"
#include "clang/Format/Format.h"
#include "clang/Frontend/ASTConsumers.h"
#include "clang/Tooling/FixIt.h"
#include <algorithm>

using namespace clang::ast_matchers;
using namespace clang;

namespace clang::tidy::cir {

Lifetime::Lifetime(StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context), codeGenOpts(Context->getCodeGenOpts()),
      cirOpts{} {
  auto OV = OptionsView(Name, Context->getOptions().CheckOptions, Context);
  codeGenOpts.ClangIRBuildDeferredThreshold =
      OV.get("CodeGenBuildDeferredThreshold", 500U);
  codeGenOpts.ClangIRSkipFunctionsFromSystemHeaders =
      OV.get("CodeGenSkipFunctionsFromSystemHeaders", false);

  cirOpts.RemarksList =
      utils::options::parseStringList(OV.get("RemarksList", ""));
  cirOpts.HistoryList =
      utils::options::parseStringList(OV.get("HistoryList", "all"));
  cirOpts.HistLimit = OV.get("HistLimit", 1U);
}

void Lifetime::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(translationUnitDecl(), this);
}

void Lifetime::setupAndRunClangIRLifetimeChecker(ASTContext &astCtx) {
  auto *TU = astCtx.getTranslationUnitDecl();
  // This is the hook used to build clangir and run the lifetime checker
  // pass. Perhaps in the future it's possible to come up with a better
  // integration story.

  // Create an instance of CIRGenerator and use it to build CIR, followed by
  // MLIR module verification.
  std::unique_ptr<::cir::CIRGenerator> Gen =
      std::make_unique<::cir::CIRGenerator>(astCtx.getDiagnostics(), nullptr,
                                            codeGenOpts);
  Gen->Initialize(astCtx);
  Gen->HandleTopLevelDecl(DeclGroupRef(TU));
  Gen->HandleTranslationUnit(astCtx);
  Gen->verifyModule();

  mlir::ModuleOp mlirMod = Gen->getModule();
  std::unique_ptr<mlir::MLIRContext> mlirCtx = Gen->takeContext();

  mlir::OpPrintingFlags flags;
  flags.enableDebugInfo(/*prettyForm=*/false);

  clang::SourceManager &clangSrcMgr = astCtx.getSourceManager();
  FileID MainFileID = clangSrcMgr.getMainFileID();

  // Do some big dance with diagnostics here: hijack clang's diagnostics with
  // MLIR one.
  llvm::MemoryBufferRef MainFileBuf = clangSrcMgr.getBufferOrFake(MainFileID);
  std::unique_ptr<llvm::MemoryBuffer> FileBuf =
      llvm::MemoryBuffer::getMemBuffer(MainFileBuf);

  llvm::SourceMgr llvmSrcMgr;
  llvmSrcMgr.AddNewSourceBuffer(std::move(FileBuf), llvm::SMLoc());

  class CIRTidyDiagnosticHandler : public mlir::SourceMgrDiagnosticHandler {
    ClangTidyCheck &tidyCheck;
    clang::SourceManager &clangSrcMgr;

    clang::SourceLocation getClangFromFileLineCol(mlir::FileLineColLoc loc) {
      clang::SourceLocation clangLoc;
      FileManager &fileMgr = clangSrcMgr.getFileManager();
      assert(loc && "not a valid mlir::FileLineColLoc");
      // The column and line may be zero to represent unknown column
      // and/or unknown line/column information.
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
      tidyCheck.diag(clangBeginLoc, diag.str(),
                     translateToClangDiagLevel(diag.getSeverity()));
      for (const auto &note : diag.getNotes()) {
        auto clangNoteBeginLoc = getClangSrcLoc(note.getLocation());
        tidyCheck.diag(clangNoteBeginLoc, note.str(),
                       translateToClangDiagLevel(note.getSeverity()));
      }
    }

    CIRTidyDiagnosticHandler(llvm::SourceMgr &mgr, mlir::MLIRContext *ctx,
                             ClangTidyCheck &tidyCheck,
                             clang::SourceManager &clangMgr,
                             ShouldShowLocFn &&shouldShowLocFn = {})
        : SourceMgrDiagnosticHandler(mgr, ctx, llvm::errs(),
                                     std::move(shouldShowLocFn)),
          tidyCheck(tidyCheck), clangSrcMgr(clangMgr) {
      setHandler(
          [this](mlir::Diagnostic &diag) { emitClangTidyDiagnostic(diag); });
    }
    ~CIRTidyDiagnosticHandler() = default;
  };

  // Use a custom diagnostic handler that can allow both regular printing
  // to stderr but also populates clang-tidy context with diagnostics (and
  // allow for instance, diagnostics to be later converted to YAML).
  CIRTidyDiagnosticHandler sourceMgrHandler(llvmSrcMgr, mlirCtx.get(), *this,
                                            clangSrcMgr);

  mlir::PassManager pm(mlirCtx.get());

  // Add pre-requisite passes to the pipeline
  pm.addPass(mlir::createMergeCleanupsPass());

  // Insert the lifetime checker.
  pm.addPass(mlir::createLifetimeCheckPass(
      cirOpts.RemarksList, cirOpts.HistoryList, cirOpts.HistLimit, &astCtx));

  bool passResult = !mlir::failed(pm.run(mlirMod));
  if (!passResult)
    llvm::report_fatal_error(
        "The pass manager failed to run pass on the module!");
}

void Lifetime::check(const MatchFinder::MatchResult &Result) {
  setupAndRunClangIRLifetimeChecker(*Result.Context);
}

} // namespace clang::tidy::cir
