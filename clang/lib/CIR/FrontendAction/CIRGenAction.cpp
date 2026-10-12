//===--- CIRGenAction.cpp - LLVM Code generation Frontend Action ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/FrontendAction/CIRGenAction.h"
#include "CIRDiagnosticHandler.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "clang/AST/ASTContext.h"
#include "clang/Basic/DiagnosticCodeGen.h"
#include "clang/Basic/DiagnosticFrontend.h"
#include "clang/CIR/CIRGenerator.h"
#include "clang/CIR/CIRToCIRPasses.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/InitAllDialects.h"
#include "clang/CIR/LowerToLLVM.h"
#include "clang/CodeGen/BackendUtil.h"
#include "clang/CodeGen/ModuleLinker.h"
#include "clang/CodeGenUtils/BackendDiagnosticHandler.h"
#include "clang/Frontend/CompilerInstance.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Frontend/Offloading/OffloadWrapper.h"
#include "llvm/IR/DiagnosticHandler.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/IPO/Internalize.h"

using namespace cir;
using namespace clang;

namespace cir {

static BackendAction
getBackendActionFromOutputType(CIRGenAction::OutputType Action) {
  switch (Action) {
  case CIRGenAction::OutputType::EmitCIR:
  case CIRGenAction::OutputType::EmitCIRBC:
    assert(false &&
           "Unsupported output type for getBackendActionFromOutputType!");
    break; // Unreachable, but fall through to report that
  case CIRGenAction::OutputType::EmitAssembly:
    return BackendAction::Backend_EmitAssembly;
  case CIRGenAction::OutputType::EmitBC:
    return BackendAction::Backend_EmitBC;
  case CIRGenAction::OutputType::EmitLLVM:
    return BackendAction::Backend_EmitLL;
  case CIRGenAction::OutputType::EmitObj:
    return BackendAction::Backend_EmitObj;
  }
  // We should only get here if a non-enum value is passed in or we went through
  // the assert(false) case above
  llvm_unreachable("Unsupported output type!");
}

static std::unique_ptr<llvm::Module>
lowerFromCIRToLLVMIR(mlir::ModuleOp MLIRModule, llvm::LLVMContext &LLVMCtx,
                     bool EnableOpenMP,
                     llvm::StringRef mlirSaveTempsOutFile = {},
                     llvm::vfs::FileSystem *fs = nullptr) {
  return direct::lowerDirectlyFromCIRToLLVMIR(MLIRModule, LLVMCtx, EnableOpenMP,
                                              mlirSaveTempsOutFile, fs);
}

// Print \p MLIRModule the way -emit-cir does, so that CIR emitted from source
// and CIR printed back from ClangIR input use the same form.
static void printCIRModule(mlir::ModuleOp MLIRModule, raw_ostream &OS) {
  mlir::OpPrintingFlags Flags;
  Flags.enableDebugInfo(/*enable=*/true, /*prettyForm=*/false);
  MLIRModule->print(OS, Flags);
}

class CIRGenConsumer : public clang::ASTConsumer {

  virtual void anchor();

  CIRGenAction::OutputType Action;

  CompilerInstance &CI;

  std::unique_ptr<raw_pwrite_stream> OutputStream;

  ASTContext *Context{nullptr};
  IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS;
  std::unique_ptr<CIRGenerator> Gen;
  const FrontendOptions &FEOptions;
  CodeGenOptions &CGO;

  llvm::LLVMContext &LLVMCtx;
  SmallVectorImpl<::clang::LinkModule> &LinkModules;

  std::optional<CIRDiagnosticHandler> MLIRDiagHandler;

  // Translates LLVM backend diagnostics (raised while lowering CIR to LLVM
  // IR and while running emitBackendOutput) into clang diagnostics; shared
  // with classic CodeGen's BackendConsumer.
  BackendDiagnosticConsumer DiagConsumer;

public:
  CIRGenConsumer(CIRGenAction::OutputType Action, CompilerInstance &CI,
                 CodeGenOptions &CGO, std::unique_ptr<raw_pwrite_stream> OS,
                 llvm::LLVMContext &LLVMCtx,
                 SmallVectorImpl<::clang::LinkModule> &LinkModules)
      : Action(Action), CI(CI), OutputStream(std::move(OS)),
        FS(&CI.getVirtualFileSystem()),
        Gen(std::make_unique<CIRGenerator>(CI.getDiagnostics(), std::move(FS),
                                           CI.getCodeGenOpts())),
        FEOptions(CI.getFrontendOpts()), CGO(CGO), LLVMCtx(LLVMCtx),
        LinkModules(LinkModules),
        DiagConsumer(CI.getDiagnostics(), CI.getCodeGenOpts()) {}

  void Initialize(ASTContext &Ctx) override {
    assert(!Context && "initialized multiple times");
    Context = &Ctx;
    DiagConsumer.setSourceManager(&Ctx.getSourceManager());
    Gen->Initialize(Ctx);
    // Install the MLIR diagnostic handler now that CIRGenerator owns its
    // MLIRContext. Lifetime is tied to this consumer, which spans CIRGen,
    // CIR-to-CIR passes, and CIR-to-LLVM lowering.
    MLIRDiagHandler.emplace(&Gen->getMLIRContext(), CI.getDiagnostics(),
                            CI.getSourceManager(), CI.getFileManager());
  }

  bool HandleTopLevelDecl(DeclGroupRef D) override {
    Gen->HandleTopLevelDecl(D);
    return true;
  }

  void HandleCXXStaticMemberVarInstantiation(clang::VarDecl *VD) override {
    Gen->HandleCXXStaticMemberVarInstantiation(VD);
  }

  void HandleOpenACCRoutineReference(const FunctionDecl *FD,
                                     const OpenACCRoutineDecl *RD) override {
    Gen->HandleOpenACCRoutineReference(FD, RD);
  }

  void HandleInlineFunctionDefinition(FunctionDecl *D) override {
    Gen->HandleInlineFunctionDefinition(D);
  }

  void HandleTranslationUnit(ASTContext &C) override {
    Gen->HandleTranslationUnit(C);

    if (!FEOptions.ClangIRDisableCIRVerifier) {
      if (!Gen->verifyModule()) {
        // Verifier output already routed through ClangIRDiagnosticHandler.
        // Only emit the generic fatal if nothing more specific was reported.
        if (!CI.getDiagnostics().hasErrorOccurred())
          CI.getDiagnostics().Report(
              diag::err_cir_verification_failed_pre_passes);
        llvm::report_fatal_error(
            "CIR codegen: module verification error before running CIR passes");
        return;
      }
    }

    mlir::ModuleOp MlirModule = Gen->getModule();
    mlir::MLIRContext &MlirCtx = Gen->getMLIRContext();

    if (!FEOptions.ClangIRDisablePasses) {
      std::string LibOptOptions = FEOptions.ClangIRLibOptOptions;

      // Setup and run CIR pipeline.
      const bool EnableLibOpt =
          FEOptions.ClangIRLibOptEnabled && (CGO.OptimizationLevel > 0);
      if (runCIRToCIRPasses(
              MlirModule, MlirCtx, !FEOptions.ClangIRDisableCIRVerifier,
              FEOptions.ClangIREnableIdiomRecognizer, CGO.OptimizationLevel > 0,
              EnableLibOpt, LibOptOptions, FEOptions.ClangIRCallConvLowering)
              .failed()) {
        // Pass-side errors already routed through ClangIRDiagnosticHandler.
        // Skip the generic catch-all if a specific diagnostic was emitted.
        if (!CI.getDiagnostics().hasErrorOccurred())
          CI.getDiagnostics().Report(diag::err_cir_to_cir_transform_failed);
        return;
      }
    }

    switch (Action) {
    case CIRGenAction::OutputType::EmitCIR:
      if (OutputStream && MlirModule)
        printCIRModule(MlirModule, *OutputStream);
      break;
    case CIRGenAction::OutputType::EmitCIRBC:
      if (OutputStream && MlirModule &&
          failed(mlir::writeBytecodeToFile(MlirModule, *OutputStream)) &&
          !CI.getDiagnostics().hasErrorOccurred())
        CI.getDiagnostics().Report(diag::err_cir_bc_write_failed);
      break;
    case CIRGenAction::OutputType::EmitLLVM:
    case CIRGenAction::OutputType::EmitBC:
    case CIRGenAction::OutputType::EmitObj:
    case CIRGenAction::OutputType::EmitAssembly: {
      StringRef saveTempsPrefix = CGO.SaveTempsFilePrefix;
      std::string cirSaveTempsOutFile, mlirSaveTempsOutFile;
      if (!saveTempsPrefix.empty()) {
        SmallString<128> stem(saveTempsPrefix);
        llvm::sys::path::replace_extension(stem, "cir");
        cirSaveTempsOutFile = std::string(stem);
        llvm::sys::path::replace_extension(stem, "mlir");
        mlirSaveTempsOutFile = std::string(stem);
      }

      if (!cirSaveTempsOutFile.empty()) {
        std::error_code ec;
        llvm::raw_fd_ostream out(cirSaveTempsOutFile, ec);
        if (!ec)
          MlirModule->print(out);
      }

      // If errors occurred during codegen, stop before running the backend.
      if (CI.getDiagnostics().hasErrorOccurred())
        return;

      // Route LLVM backend diagnostics (optimization remarks, unsupported
      // features, inline-asm errors, etc.) through clang diagnostics for
      // the remainder of the LLVM-emitting pipeline.
      std::unique_ptr<llvm::DiagnosticHandler> OldDiagnosticHandler =
          LLVMCtx.getDiagnosticHandler();
      llvm::scope_exit RestoreDiagnosticHandler([&]() {
        LLVMCtx.setDiagnosticHandler(std::move(OldDiagnosticHandler));
      });
      LLVMCtx.setDiagnosticHandler(DiagConsumer.createDiagnosticHandler());

      std::unique_ptr<llvm::Module> LLVMModule = lowerFromCIRToLLVMIR(
          MlirModule, LLVMCtx, C.getLangOpts().OpenMP, mlirSaveTempsOutFile,
          &CI.getVirtualFileSystem());

      LLVMModule->setDataLayout(C.getTargetInfo().getDataLayoutString());

      for (llvm::Function &F : LLVMModule->functions())
        if (const Decl *FD = Gen->getDeclForMangledName(F.getName()))
          DiagConsumer.addFunctionSourceLocation(
              F.getName(), FD->getASTContext().getFullLoc(FD->getLocation()));

      if (linkInModules(*LLVMModule))
        return;

      // Embed the offloaded SYCL device binary into the host module.
      if (C.getLangOpts().SYCLIsHost && !CGO.OffloadBinaryToEmbedFile.empty())
        embedSYCLDeviceBinary(*LLVMModule);

      // CUDA, HIP and OpenMP offloading rely on host-side offload entries that
      // are not emitted on the ClangIR path yet, so embedding their device
      // objects would produce a host object that cannot be registered.
      const LangOptions &LangOpts = C.getLangOpts();
      if (!CGO.OffloadObjects.empty() &&
          (LangOpts.CUDA || !LangOpts.OMPTargetTriples.empty())) {
        DiagnosticsEngine &Diags = CI.getDiagnostics();
        Diags.Report(Diags.getCustomDiagID(
            DiagnosticsEngine::Error,
            "ClangIR code gen Not Yet Implemented: embedding offload objects "
            "for CUDA, HIP or OpenMP offloading"));
        return;
      }

      // If there is device offloading code embed it in the host now.
      EmbedObject(LLVMModule.get(), CGO, CI.getVirtualFileSystem(),
                  CI.getDiagnostics());

      BackendAction BEAction = getBackendActionFromOutputType(Action);
      emitBackendOutput(CI, CI.getCodeGenOpts(), LLVMModule.get(), BEAction, FS,
                        std::move(OutputStream));
      break;
    }
    }
  }

  // TODO: share with BackendConsumer::LinkInModules once the rest of the
  // linking logic (not just diagnostics) is unified.
  bool linkInModules(llvm::Module &M) {
    for (auto &LM : LinkModules) {
      assert(LM.Module && "LinkModule does not actually have a module");

      if (LM.PropagateAttrs)
        for (llvm::Function &F : *LM.Module) {
          if (F.isIntrinsic())
            continue;
          clang::CodeGen::mergeDefaultFunctionDefinitionAttributes(
              F, CGO, CI.getLangOpts(), CI.getTargetOpts(), LM.Internalize);
        }

      DiagConsumer.setCurLinkModule(LM.Module.get());
      bool Err;
      if (LM.Internalize) {
        Err = llvm::Linker::linkModules(
            M, std::move(LM.Module), LM.LinkFlags,
            [](llvm::Module &M, const llvm::StringSet<> &GVS) {
              llvm::internalizeModule(M, [&GVS](const llvm::GlobalValue &GV) {
                return !GV.hasName() || (GVS.count(GV.getName()) == 0);
              });
            });
      } else {
        Err = llvm::Linker::linkModules(M, std::move(LM.Module), LM.LinkFlags);
      }

      if (Err)
        return true;
    }

    LinkModules.clear();
    return false;
  }

  // Reads the device binary named by -foffload-include-binary and embeds it
  // into the host module. wrapSYCLBinaries also appends the registration ctor
  // at priority 101 when no registration-function out-param is supplied.
  void embedSYCLDeviceBinary(llvm::Module &M) {
    StringRef fileName = CGO.OffloadBinaryToEmbedFile;
    auto bufferOrErr = CI.getVirtualFileSystem().getBufferForFile(fileName);
    if (std::error_code ec = bufferOrErr.getError()) {
      CI.getDiagnostics().Report(diag::err_cannot_open_file)
          << fileName << ec.message();
      return;
    }
    std::unique_ptr<llvm::MemoryBuffer> buffer = std::move(bufferOrErr.get());
    if (llvm::Error err = llvm::offloading::wrapSYCLBinaries(
            M,
            ArrayRef<char>(buffer->getBufferStart(), buffer->getBufferSize()),
            llvm::offloading::SYCLJITOptions(), /*IsFinalizedImage=*/true)) {
      CI.getDiagnostics().Report(diag::err_fe_error_backend)
          << llvm::toString(std::move(err));
      return;
    }
  }

  void HandleTagDeclDefinition(TagDecl *D) override {
    PrettyStackTraceDecl CrashInfo(D, SourceLocation(),
                                   Context->getSourceManager(),
                                   "CIR generation of declaration");
    Gen->HandleTagDeclDefinition(D);
  }

  void HandleTagDeclRequiredDefinition(const TagDecl *D) override {
    Gen->HandleTagDeclRequiredDefinition(D);
  }

  void CompleteTentativeDefinition(VarDecl *D) override {
    Gen->CompleteTentativeDefinition(D);
  }

  void HandleVTable(CXXRecordDecl *RD) override { Gen->HandleVTable(RD); }
};
} // namespace cir

void CIRGenConsumer::anchor() {}

CIRGenAction::CIRGenAction(OutputType Act, mlir::MLIRContext *MLIRCtx)
    : MLIRCtx(MLIRCtx ? MLIRCtx : new mlir::MLIRContext),
      Ctx(std::make_unique<llvm::LLVMContext>()), Action(Act) {}

CIRGenAction::~CIRGenAction() { MLIRMod.release(); }

bool CIRGenAction::BeginSourceFileAction(CompilerInstance &CI) {
  if (clang::loadLinkModules(CI, *Ctx, LinkModules))
    return false;
  return ASTFrontendAction::BeginSourceFileAction(CI);
}

static std::unique_ptr<raw_pwrite_stream>
getOutputStream(CompilerInstance &CI, StringRef InFile,
                CIRGenAction::OutputType Action);

void CIRGenAction::ExecuteAction() {
  if (getCurrentFileKind().getLanguage() != Language::CIR) {
    ASTFrontendAction::ExecuteAction();
    return;
  }

  CompilerInstance &CI = getCompilerInstance();
  DiagnosticsEngine &Diags = CI.getDiagnostics();
  SourceManager &SM = CI.getSourceManager();

  std::unique_ptr<raw_pwrite_stream> OS = CI.takeOutputStream();
  if (!OS)
    OS = getOutputStream(CI, getCurrentFileOrBufferName(), Action);
  if (!OS)
    return;

  std::optional<llvm::MemoryBufferRef> MainFile =
      SM.getBufferOrNone(SM.getMainFileID());
  if (!MainFile)
    return;

  mlir::MLIRContext MLIRContext;
  cir::registerAllDialects(MLIRContext);

  // Route parser and verifier errors through clang's diagnostics. Parse
  // errors point into the .cir input. Verifier errors use the location of
  // the failing op, which for CIR emitted by CIRGen is a location in the
  // original source file; that file is not loaded in the SourceManager, so
  // the handler reports such errors with the original location as text.
  // TODO: Decide where errors in CIRGen-produced input should point: at the
  // .cir text, or at the original source.
  CIRDiagnosticHandler DiagHandler(&MLIRContext, Diags, SM,
                                   CI.getFileManager());

  mlir::OwningOpRef<mlir::ModuleOp> Module =
      mlir::parseSourceString<mlir::ModuleOp>(MainFile->getBuffer(),
                                              mlir::ParserConfig(&MLIRContext),
                                              MainFile->getBufferIdentifier());
  if (!Module) {
    if (!Diags.hasErrorOccurred())
      Diags.Report(diag::err_invalid_cir);
    return;
  }

  // CIR is lowered for the target ABI of its own triple, so a module emitted
  // for a different target cannot be retargeted by overriding its triple.
  auto ModuleTriple = mlir::dyn_cast_if_present<mlir::StringAttr>(
      (*Module)->getAttr(cir::CIRDialect::getTripleAttrName()));
  if (!ModuleTriple) {
    Diags.Report(diag::err_cir_input_missing_triple);
    return;
  }
  const std::string &TargetTriple = CI.getTarget().getTriple().str();
  if (ModuleTriple.getValue() != TargetTriple) {
    Diags.Report(diag::err_cir_input_triple_mismatch)
        << ModuleTriple.getValue() << TargetTriple;
    return;
  }

  switch (Action) {
  case OutputType::EmitCIR:
    // The CIR-to-CIR pipeline is not run: CIR printed by -emit-cir has already
    // been through it, and its lowering passes are not idempotent.
    printCIRModule(*Module, *OS);
    break;
  case OutputType::EmitCIRBC:
    if (failed(mlir::writeBytecodeToFile(*Module, *OS)) &&
        !Diags.hasErrorOccurred())
      Diags.Report(diag::err_cir_bc_write_failed);
    break;
  case OutputType::EmitLLVM:
  case OutputType::EmitBC:
  case OutputType::EmitObj:
  case OutputType::EmitAssembly:
    Diags.Report(diag::err_fe_cir_input_unsupported);
    break;
  }
}

static std::unique_ptr<raw_pwrite_stream>
getOutputStream(CompilerInstance &CI, StringRef InFile,
                CIRGenAction::OutputType Action) {
  switch (Action) {
  case CIRGenAction::OutputType::EmitAssembly:
    return CI.createDefaultOutputFile(false, InFile, "s");
  case CIRGenAction::OutputType::EmitCIR:
    return CI.createDefaultOutputFile(false, InFile, "cir");
  case CIRGenAction::OutputType::EmitCIRBC:
    return CI.createDefaultOutputFile(true, InFile, "cirbc");
  case CIRGenAction::OutputType::EmitLLVM:
    return CI.createDefaultOutputFile(false, InFile, "ll");
  case CIRGenAction::OutputType::EmitBC:
    return CI.createDefaultOutputFile(true, InFile, "bc");
  case CIRGenAction::OutputType::EmitObj:
    return CI.createDefaultOutputFile(true, InFile, "o");
  }
  llvm_unreachable("Invalid CIRGenAction::OutputType");
}

std::unique_ptr<ASTConsumer>
CIRGenAction::CreateASTConsumer(CompilerInstance &CI, StringRef InFile) {
  std::unique_ptr<llvm::raw_pwrite_stream> Out = CI.takeOutputStream();

  if (!Out)
    Out = getOutputStream(CI, InFile, Action);

  auto Result = std::make_unique<cir::CIRGenConsumer>(
      Action, CI, CI.getCodeGenOpts(), std::move(Out), *Ctx, LinkModules);

  return Result;
}

void EmitAssemblyAction::anchor() {}
EmitAssemblyAction::EmitAssemblyAction(mlir::MLIRContext *MLIRCtx)
    : CIRGenAction(OutputType::EmitAssembly, MLIRCtx) {}

void EmitCIRAction::anchor() {}
EmitCIRAction::EmitCIRAction(mlir::MLIRContext *MLIRCtx)
    : CIRGenAction(OutputType::EmitCIR, MLIRCtx) {}

void EmitCIRBCAction::anchor() {}
EmitCIRBCAction::EmitCIRBCAction(mlir::MLIRContext *MLIRCtx)
    : CIRGenAction(OutputType::EmitCIRBC, MLIRCtx) {}

void EmitLLVMAction::anchor() {}
EmitLLVMAction::EmitLLVMAction(mlir::MLIRContext *MLIRCtx)
    : CIRGenAction(OutputType::EmitLLVM, MLIRCtx) {}

void EmitBCAction::anchor() {}
EmitBCAction::EmitBCAction(mlir::MLIRContext *MLIRCtx)
    : CIRGenAction(OutputType::EmitBC, MLIRCtx) {}

void EmitObjAction::anchor() {}
EmitObjAction::EmitObjAction(mlir::MLIRContext *MLIRCtx)
    : CIRGenAction(OutputType::EmitObj, MLIRCtx) {}
