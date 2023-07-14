//===--- CIRGenAction.cpp - LLVM Code generation Frontend Action ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CIRFrontendAction/CIRGenAction.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Parser/Parser.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclGroup.h"
#include "clang/Basic/DiagnosticFrontend.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/LangStandard.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/CIR/CIRGenerator.h"
#include "clang/CIR/CIRToCIRPasses.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/LowerToLLVM.h"
#include "clang/CIR/Passes.h"
#include "clang/CodeGen/BackendUtil.h"
#include "clang/CodeGen/ModuleBuilder.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LLVMRemarkStreamer.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/LTO/LTOBackend.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Pass.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TimeProfiler.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Transforms/IPO/Internalize.h"

#include <memory>

using namespace cir;
using namespace clang;

static std::string sanitizePassOptions(llvm::StringRef o) {
  std::string opts{o};
  // MLIR pass options are space separated, but we use ';' in clang since
  // space aren't well supported, switch it back.
  for (unsigned i = 0, e = opts.size(); i < e; ++i)
    if (opts[i] == ';')
      opts[i] = ' ';
  // If arguments are surrounded with '"', trim them off
  return llvm::StringRef(opts).trim('"').str();
}

namespace cir {

static std::unique_ptr<llvm::Module> lowerFromCIRToLLVMIR(
    const clang::FrontendOptions &feOptions, mlir::ModuleOp mlirMod,
    std::unique_ptr<mlir::MLIRContext> mlirCtx, llvm::LLVMContext &llvmCtx) {
  if (feOptions.ClangIRDirectLowering)
    return direct::lowerDirectlyFromCIRToLLVMIR(mlirMod, llvmCtx);
  else
    return lowerFromCIRToMLIRToLLVMIR(mlirMod, std::move(mlirCtx), llvmCtx);
}

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

  void Initialize(ASTContext &ctx) override {
    assert(!astContext && "initialized multiple times");

    astContext = &ctx;

    gen->Initialize(ctx);
  }

  bool HandleTopLevelDecl(DeclGroupRef D) override {
    PrettyStackTraceDecl CrashInfo(*D.begin(), SourceLocation(),
                                   astContext->getSourceManager(),
                                   "LLVM IR generation of declaration");
    gen->HandleTopLevelDecl(D);
    return true;
  }

  void HandleCXXStaticMemberVarInstantiation(clang::VarDecl *VD) override {
    gen->HandleCXXStaticMemberVarInstantiation(VD);
  }

  void HandleInlineFunctionDefinition(FunctionDecl *D) override {
    gen->HandleInlineFunctionDefinition(D);
  }

  void HandleInterestingDecl(DeclGroupRef D) override {
    llvm_unreachable("NYI");
  }

  void HandleTranslationUnit(ASTContext &C) override {
    // Note that this method is called after `HandleTopLevelDecl` has already
    // ran all over the top level decls. Here clang mostly wraps defered and
    // global codegen, followed by running CIR passes.
    gen->HandleTranslationUnit(C);

    if (!feOptions.ClangIRDisableCIRVerifier)
      if (!gen->verifyModule()) {
        llvm::report_fatal_error(
            "CIR codegen: module verification error before running CIR passes");
        return;
      }

    auto mlirMod = gen->getModule();
    auto mlirCtx = gen->takeContext();

    auto setupCIRPipelineAndExecute = [&] {
      // Sanitize passes options. MLIR uses spaces between pass options
      // and since that's hard to fly in clang, we currently use ';'.
      std::string lifetimeOpts;
      if (feOptions.ClangIRLifetimeCheck)
        lifetimeOpts = sanitizePassOptions(feOptions.ClangIRLifetimeCheckOpts);

      // Setup and run CIR pipeline.
      bool passOptParsingFailure = false;
      if (runCIRToCIRPasses(mlirMod, mlirCtx.get(), C,
                            !feOptions.ClangIRDisableCIRVerifier,
                            feOptions.ClangIRLifetimeCheck, lifetimeOpts,
                            passOptParsingFailure)
              .failed()) {
        if (passOptParsingFailure)
          diagnosticsEngine.Report(diag::err_drv_cir_pass_opt_parsing)
              << feOptions.ClangIRLifetimeCheckOpts;
        else
          llvm::report_fatal_error("CIR codegen: MLIR pass manager fails "
                                   "when running CIR passes!");
        return;
      }
    };

    if (!feOptions.ClangIRDisablePasses) {
      // Handle source manager properly given that lifetime analysis
      // might emit warnings and remarks.
      auto &clangSourceMgr = C.getSourceManager();
      FileID MainFileID = clangSourceMgr.getMainFileID();

      std::unique_ptr<llvm::MemoryBuffer> FileBuf =
          llvm::MemoryBuffer::getMemBuffer(
              clangSourceMgr.getBufferOrFake(MainFileID));

      llvm::SourceMgr mlirSourceMgr;
      mlirSourceMgr.AddNewSourceBuffer(std::move(FileBuf), llvm::SMLoc());

      if (feOptions.ClangIRVerifyDiags) {
        mlir::SourceMgrDiagnosticVerifierHandler sourceMgrHandler(
            mlirSourceMgr, mlirCtx.get());
        mlirCtx->printOpOnDiagnostic(false);
        setupCIRPipelineAndExecute();

        // Verify the diagnostic handler to make sure that each of the
        // diagnostics matched.
        if (sourceMgrHandler.verify().failed()) {
          // FIXME: we fail ungracefully, there's probably a better way
          // to communicate non-zero return so tests can actually fail.
          llvm::sys::RunInterruptHandlers();
          exit(1);
        }
      } else {
        mlir::SourceMgrDiagnosticHandler sourceMgrHandler(mlirSourceMgr,
                                                          mlirCtx.get());
        setupCIRPipelineAndExecute();
      }
    }

    switch (action) {
    case CIRGenAction::OutputType::EmitCIR:
      if (outputStream && mlirMod) {
        // Emit remaining defaulted C++ methods
        if (!feOptions.ClangIRDisableEmitCXXDefault)
          gen->buildDefaultMethods();

        // FIXME: we cannot roundtrip prettyForm=true right now.
        mlir::OpPrintingFlags flags;
        flags.enableDebugInfo(/*enable=*/true, /*prettyForm=*/false);
        mlirMod->print(*outputStream, flags);
      }
      break;
    case CIRGenAction::OutputType::EmitMLIR: {
      auto loweredMlirModule = lowerFromCIRToMLIR(mlirMod, mlirCtx.get());
      assert(outputStream && "Why are we here without an output stream?");
      // FIXME: we cannot roundtrip prettyForm=true right now.
      mlir::OpPrintingFlags flags;
      flags.enableDebugInfo(/*enable=*/true, /*prettyForm=*/false);
      loweredMlirModule->print(*outputStream, flags);
      break;
    }
    case CIRGenAction::OutputType::EmitLLVM: {
      llvm::LLVMContext llvmCtx;
      auto llvmModule =
          lowerFromCIRToLLVMIR(feOptions, mlirMod, std::move(mlirCtx), llvmCtx);

      llvmModule->setTargetTriple(targetOptions.Triple);

      EmitBackendOutput(diagnosticsEngine, headerSearchOptions, codeGenOptions,
                        targetOptions, langOptions,
                        C.getTargetInfo().getDataLayoutString(),
                        llvmModule.get(), BackendAction::Backend_EmitLL, FS,
                        std::move(outputStream));
      break;
    }
    case CIRGenAction::OutputType::EmitObj: {
      llvm::LLVMContext llvmCtx;
      auto llvmModule =
          lowerFromCIRToLLVMIR(feOptions, mlirMod, std::move(mlirCtx), llvmCtx);

      llvmModule->setTargetTriple(targetOptions.Triple);
      EmitBackendOutput(diagnosticsEngine, headerSearchOptions, codeGenOptions,
                        targetOptions, langOptions,
                        C.getTargetInfo().getDataLayoutString(),
                        llvmModule.get(), BackendAction::Backend_EmitObj, FS,
                        std::move(outputStream));
      break;
    }
    case CIRGenAction::OutputType::EmitAssembly: {
      llvm::LLVMContext llvmCtx;
      auto llvmModule =
          lowerFromCIRToLLVMIR(feOptions, mlirMod, std::move(mlirCtx), llvmCtx);

      llvmModule->setTargetTriple(targetOptions.Triple);
      EmitBackendOutput(diagnosticsEngine, headerSearchOptions, codeGenOptions,
                        targetOptions, langOptions,
                        C.getTargetInfo().getDataLayoutString(),
                        llvmModule.get(), BackendAction::Backend_EmitAssembly,
                        FS, std::move(outputStream));
      break;
    }
    case CIRGenAction::OutputType::None:
      break;
    }
  }

  void HandleTagDeclDefinition(TagDecl *D) override {
    PrettyStackTraceDecl CrashInfo(D, SourceLocation(),
                                   astContext->getSourceManager(),
                                   "CIR generation of declaration");
    gen->HandleTagDeclDefinition(D);
  }

  void HandleTagDeclRequiredDefinition(const TagDecl *D) override {
    gen->HandleTagDeclRequiredDefinition(D);
  }

  void CompleteTentativeDefinition(VarDecl *D) override {
    gen->CompleteTentativeDefinition(D);
  }

  void CompleteExternalDeclaration(VarDecl *D) override {
    llvm_unreachable("NYI");
  }

  void AssignInheritanceModel(CXXRecordDecl *RD) override {
    llvm_unreachable("NYI");
  }

  void HandleVTable(CXXRecordDecl *RD) override { gen->HandleVTable(RD); }
};
} // namespace cir

void CIRGenConsumer::anchor() {}

CIRGenAction::CIRGenAction(OutputType act, mlir::MLIRContext *_MLIRContext)
    : mlirContext(_MLIRContext ? _MLIRContext : new mlir::MLIRContext),
      action(act) {}

CIRGenAction::~CIRGenAction() { mlirModule.reset(); }

void CIRGenAction::EndSourceFileAction() {
  // If the consumer creation failed, do nothing.
  if (!getCompilerInstance().hasASTConsumer())
    return;

  // TODO: pass the module around
  // module = cgConsumer->takeModule();
}

static std::unique_ptr<raw_pwrite_stream>
getOutputStream(CompilerInstance &ci, StringRef inFile,
                CIRGenAction::OutputType action) {
  switch (action) {
  case CIRGenAction::OutputType::EmitAssembly:
    return ci.createDefaultOutputFile(false, inFile, "s");
  case CIRGenAction::OutputType::EmitCIR:
    return ci.createDefaultOutputFile(false, inFile, "cir");
  case CIRGenAction::OutputType::EmitMLIR:
    return ci.createDefaultOutputFile(false, inFile, "mlir");
  case CIRGenAction::OutputType::EmitLLVM:
    return ci.createDefaultOutputFile(false, inFile, "llvm");
  case CIRGenAction::OutputType::EmitObj:
    return ci.createDefaultOutputFile(true, inFile, "o");
  case CIRGenAction::OutputType::None:
    return nullptr;
  }

  llvm_unreachable("Invalid action!");
}

std::unique_ptr<ASTConsumer>
CIRGenAction::CreateASTConsumer(CompilerInstance &ci, StringRef inputFile) {
  auto out = ci.takeOutputStream();
  if (!out)
    out = getOutputStream(ci, inputFile, action);

  auto Result = std::make_unique<cir::CIRGenConsumer>(
      action, ci.getDiagnostics(), &ci.getVirtualFileSystem(),
      ci.getHeaderSearchOpts(), ci.getCodeGenOpts(), ci.getTargetOpts(),
      ci.getLangOpts(), ci.getFrontendOpts(), std::move(out));
  cgConsumer = Result.get();

  // Enable generating macro debug info only when debug info is not disabled and
  // also macrod ebug info is enabled
  if (ci.getCodeGenOpts().getDebugInfo() != llvm::codegenoptions::NoDebugInfo &&
      ci.getCodeGenOpts().MacroDebugInfo) {
    llvm_unreachable("NYI");
  }

  return std::move(Result);
}

mlir::OwningOpRef<mlir::ModuleOp>
CIRGenAction::loadModule(llvm::MemoryBufferRef mbRef) {
  auto module =
      mlir::parseSourceString<mlir::ModuleOp>(mbRef.getBuffer(), mlirContext);
  assert(module && "Failed to parse ClangIR module");
  return module;
}

void CIRGenAction::ExecuteAction() {
  if (getCurrentFileKind().getLanguage() != Language::CIR) {
    this->ASTFrontendAction::ExecuteAction();
    return;
  }

  // If this is a CIR file we have to treat it specially.
  // TODO: This could be done more logically. This is just modeled at the moment
  // mimicing CodeGenAction but this is clearly suboptimal.
  auto &ci = getCompilerInstance();
  std::unique_ptr<raw_pwrite_stream> outstream =
      getOutputStream(ci, getCurrentFile(), action);
  if (action != OutputType::None && !outstream)
    return;

  auto &sourceManager = ci.getSourceManager();
  auto fileID = sourceManager.getMainFileID();
  auto mainFile = sourceManager.getBufferOrNone(fileID);

  if (!mainFile)
    return;

  mlirContext->getOrLoadDialect<mlir::cir::CIRDialect>();
  mlirContext->getOrLoadDialect<mlir::func::FuncDialect>();
  mlirContext->getOrLoadDialect<mlir::memref::MemRefDialect>();

  // TODO: unwrap this -- this exists because including the `OwningModuleRef` in
  // CIRGenAction's header would require linking the Frontend against MLIR.
  // Let's avoid that for now.
  auto mlirModule = loadModule(*mainFile);
  if (!mlirModule)
    return;

  llvm::LLVMContext llvmCtx;
  auto llvmModule = lowerFromCIRToLLVMIR(
      ci.getFrontendOpts(), mlirModule.release(),
      std::unique_ptr<mlir::MLIRContext>(mlirContext), llvmCtx);

  if (outstream)
    llvmModule->print(*outstream, nullptr);
}

void EmitAssemblyAction::anchor() {}
EmitAssemblyAction::EmitAssemblyAction(mlir::MLIRContext *_MLIRContext)
    : CIRGenAction(OutputType::EmitAssembly, _MLIRContext) {}

void EmitCIRAction::anchor() {}
EmitCIRAction::EmitCIRAction(mlir::MLIRContext *_MLIRContext)
    : CIRGenAction(OutputType::EmitCIR, _MLIRContext) {}

void EmitCIROnlyAction::anchor() {}
EmitCIROnlyAction::EmitCIROnlyAction(mlir::MLIRContext *_MLIRContext)
    : CIRGenAction(OutputType::None, _MLIRContext) {}

void EmitMLIRAction::anchor() {}
EmitMLIRAction::EmitMLIRAction(mlir::MLIRContext *_MLIRContext)
    : CIRGenAction(OutputType::EmitMLIR, _MLIRContext) {}

void EmitLLVMAction::anchor() {}
EmitLLVMAction::EmitLLVMAction(mlir::MLIRContext *_MLIRContext)
    : CIRGenAction(OutputType::EmitLLVM, _MLIRContext) {}

void EmitObjAction::anchor() {}
EmitObjAction::EmitObjAction(mlir::MLIRContext *_MLIRContext)
    : CIRGenAction(OutputType::EmitObj, _MLIRContext) {}
