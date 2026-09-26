//===--- CodeGenAction.cpp - LLVM Code Generation Frontend Action ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CodeGen/CodeGenAction.h"
#include "BackendConsumer.h"
#include "CGCall.h"
#include "CodeGenModule.h"
#include "CoverageMappingGen.h"
#include "MacroPPCallbacks.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclGroup.h"
#include "clang/Basic/DiagnosticFrontend.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/LangStandard.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/CodeGen/BackendUtil.h"
#include "clang/CodeGen/ModuleBuilder.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/MultiplexConsumer.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Serialization/ASTWriter.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/CodeGen/MachineOptimizationRemarkEmitter.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LLVMRemarkStreamer.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassTimingInfo.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/LTO/LTOBackend.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Pass.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Mutex.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TimeProfiler.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Transforms/IPO/Internalize.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <optional>
using namespace clang;
using namespace llvm;

#define DEBUG_TYPE "codegenaction"

namespace {
llvm::ManagedStatic<llvm::sys::SmartMutex<true>> TimePassesMutex;
}

namespace clang {

static void reportOptRecordError(Error E, DiagnosticsEngine &Diags,
                                 const CodeGenOptions &CodeGenOpts) {
  handleAllErrors(
      std::move(E),
    [&](const LLVMRemarkSetupFileError &E) {
        Diags.Report(diag::err_cannot_open_file)
            << CodeGenOpts.OptRecordFile << E.message();
      },
    [&](const LLVMRemarkSetupPatternError &E) {
        Diags.Report(diag::err_drv_optimization_remark_pattern)
            << E.message() << CodeGenOpts.OptRecordPasses;
      },
    [&](const LLVMRemarkSetupFormatError &E) {
        Diags.Report(diag::err_drv_optimization_remark_format)
            << CodeGenOpts.OptRecordFormat;
      });
}

BackendConsumer::BackendConsumer(CompilerInstance &CI, BackendAction Action,
                                 IntrusiveRefCntPtr<llvm::vfs::FileSystem> VFS,
                                 LLVMContext &C,
                                 SmallVector<LinkModule, 4> LinkModules,
                                 StringRef InFile,
                                 std::unique_ptr<raw_pwrite_stream> OS,
                                 CoverageSourceInfo *CoverageInfo,
                                 llvm::Module *CurLinkModule)
    : CI(CI), Diags(CI.getDiagnostics()), CodeGenOpts(CI.getCodeGenOpts()),
      TargetOpts(CI.getTargetOpts()), LangOpts(CI.getLangOpts()),
      AsmOutStream(std::move(OS)), FS(VFS), Action(Action),
      Gen(CreateLLVMCodeGen(CI, InFile, C, CoverageInfo)),
      LinkModules(std::move(LinkModules)), DiagConsumer(Diags, CodeGenOpts) {
  DiagConsumer.setCurLinkModule(CurLinkModule);
  TimerIsEnabled = CodeGenOpts.TimePasses;
  {
    llvm::sys::SmartScopedLock<true> Lock(*TimePassesMutex);
    llvm::TimePassesIsEnabled = CodeGenOpts.TimePasses;
    llvm::TimePassesPerRun = CodeGenOpts.TimePassesPerRun;
  }
  if (CodeGenOpts.TimePasses)
    LLVMIRGeneration.init("irgen", "LLVM IR generation", CI.getTimerGroup());
}

llvm::Module* BackendConsumer::getModule() const {
  return Gen->GetModule();
}

std::unique_ptr<llvm::Module> BackendConsumer::takeModule() {
  return std::unique_ptr<llvm::Module>(Gen->ReleaseModule());
}

CodeGenerator* BackendConsumer::getCodeGenerator() {
  return Gen.get();
}

void BackendConsumer::HandleCXXStaticMemberVarInstantiation(VarDecl *VD) {
  Gen->HandleCXXStaticMemberVarInstantiation(VD);
}

void BackendConsumer::Initialize(ASTContext &Ctx) {
  assert(!Context && "initialized multiple times");

  Context = &Ctx;
  DiagConsumer.setSourceManager(&Ctx.getSourceManager());

  if (TimerIsEnabled)
    LLVMIRGeneration.startTimer();

  Gen->Initialize(Ctx);

  if (TimerIsEnabled)
    LLVMIRGeneration.stopTimer();
}

bool BackendConsumer::HandleTopLevelDecl(DeclGroupRef D) {
  PrettyStackTraceDecl CrashInfo(*D.begin(), SourceLocation(),
                                 Context->getSourceManager(),
                                 "LLVM IR generation of declaration");

  // Recurse.
  if (TimerIsEnabled && !LLVMIRGenerationRefCount++)
    CI.getFrontendTimer().yieldTo(LLVMIRGeneration);

  Gen->HandleTopLevelDecl(D);

  if (TimerIsEnabled && !--LLVMIRGenerationRefCount)
    LLVMIRGeneration.yieldTo(CI.getFrontendTimer());

  return true;
}

void BackendConsumer::HandleInlineFunctionDefinition(FunctionDecl *D) {
  PrettyStackTraceDecl CrashInfo(D, SourceLocation(),
                                 Context->getSourceManager(),
                                 "LLVM IR generation of inline function");
  if (TimerIsEnabled)
    CI.getFrontendTimer().yieldTo(LLVMIRGeneration);

  Gen->HandleInlineFunctionDefinition(D);

  if (TimerIsEnabled)
    LLVMIRGeneration.yieldTo(CI.getFrontendTimer());
}

void BackendConsumer::HandleInterestingDecl(DeclGroupRef D) {
  HandleTopLevelDecl(D);
}

// Links each entry in LinkModules into our module. Returns true on error.
bool BackendConsumer::LinkInModules(llvm::Module *M) {
  for (auto &LM : LinkModules) {
    assert(LM.Module && "LinkModule does not actually have a module");

    if (LM.PropagateAttrs)
      for (Function &F : *LM.Module) {
        // Skip intrinsics. Keep consistent with how intrinsics are created
        // in LLVM IR.
        if (F.isIntrinsic())
          continue;
        CodeGen::mergeDefaultFunctionDefinitionAttributes(
          F, CodeGenOpts, LangOpts, TargetOpts, LM.Internalize);
      }

    DiagConsumer.setCurLinkModule(LM.Module.get());
    bool Err;

    if (LM.Internalize) {
      Err = Linker::linkModules(
          *M, std::move(LM.Module), LM.LinkFlags,
          [](llvm::Module &M, const llvm::StringSet<> &GVS) {
            internalizeModule(M, [&GVS](const llvm::GlobalValue &GV) {
              return !GV.hasName() || (GVS.count(GV.getName()) == 0);
            });
          });
    } else
      Err = Linker::linkModules(*M, std::move(LM.Module), LM.LinkFlags);

    if (Err)
      return true;
  }

  LinkModules.clear();
  return false; // success
}

void BackendConsumer::HandleTranslationUnit(ASTContext &C) {
  {
    llvm::TimeTraceScope TimeScope("Frontend");
    PrettyStackTraceString CrashInfo("Per-file LLVM IR generation");
    if (TimerIsEnabled && !LLVMIRGenerationRefCount++)
      CI.getFrontendTimer().yieldTo(LLVMIRGeneration);

    Gen->HandleTranslationUnit(C);

    if (TimerIsEnabled && !--LLVMIRGenerationRefCount)
      LLVMIRGeneration.yieldTo(CI.getFrontendTimer());
  }

  // Silently ignore if we weren't initialized for some reason.
  if (!getModule())
    return;

  LLVMContext &Ctx = getModule()->getContext();
  std::unique_ptr<DiagnosticHandler> OldDiagnosticHandler =
    Ctx.getDiagnosticHandler();
  llvm::scope_exit RestoreDiagnosticHandler(
      [&]() { Ctx.setDiagnosticHandler(std::move(OldDiagnosticHandler)); });
  Ctx.setDiagnosticHandler(createDiagnosticHandler());

  Ctx.setDefaultTargetCPU(TargetOpts.CPU);
  Ctx.setDefaultTargetFeatures(llvm::join(TargetOpts.Features, ","));

  Expected<LLVMRemarkFileHandle> OptRecordFileOrErr =
      setupLLVMOptimizationRemarks(
          Ctx, CodeGenOpts.OptRecordFile, CodeGenOpts.OptRecordPasses,
          CodeGenOpts.OptRecordFormat, CodeGenOpts.DiagnosticsWithHotness,
          CodeGenOpts.DiagnosticsHotnessThreshold);

  if (Error E = OptRecordFileOrErr.takeError()) {
    reportOptRecordError(std::move(E), Diags, CodeGenOpts);
    return;
  }

  LLVMRemarkFileHandle OptRecordFile = std::move(*OptRecordFileOrErr);

  if (OptRecordFile && CodeGenOpts.getProfileUse() !=
                           llvm::driver::ProfileInstrKind::ProfileNone)
    Ctx.setDiagnosticsHotnessRequested(true);

  if (CodeGenOpts.MisExpect) {
    Ctx.setMisExpectWarningRequested(true);
  }

  if (CodeGenOpts.DiagnosticsMisExpectTolerance) {
    Ctx.setDiagnosticsMisExpectTolerance(
      CodeGenOpts.DiagnosticsMisExpectTolerance);
  }

  // Link each LinkModule into our module.
  if (!CodeGenOpts.LinkBitcodePostopt && LinkInModules(getModule()))
    return;

  for (auto &F : getModule()->functions()) {
    if (const Decl *FD = Gen->GetDeclForMangledName(F.getName())) {
      auto Loc = FD->getASTContext().getFullLoc(FD->getLocation());
      DiagConsumer.addFunctionSourceLocation(F.getName(), Loc);
    }
  }

  if (CodeGenOpts.ClearASTBeforeBackend) {
    LLVM_DEBUG(llvm::dbgs() << "Clearing AST...\n");
    // Access to the AST is no longer available after this.
    // Other things that the ASTContext manages are still available, e.g.
    // the SourceManager. It'd be nice if we could separate out all the
    // things in ASTContext used after this point and null out the
    // ASTContext, but too many various parts of the ASTContext are still
    // used in various parts.
    C.cleanup();
    C.getAllocator().Reset();
  }

  EmbedBitcode(getModule(), CodeGenOpts, llvm::MemoryBufferRef());

  emitBackendOutput(CI, CI.getCodeGenOpts(), getModule(), Action, FS,
                    std::move(AsmOutStream), this);

  if (OptRecordFile)
    OptRecordFile->keep();
}

void BackendConsumer::HandleTagDeclDefinition(TagDecl *D) {
  PrettyStackTraceDecl CrashInfo(D, SourceLocation(),
                                 Context->getSourceManager(),
                                 "LLVM IR generation of declaration");
  Gen->HandleTagDeclDefinition(D);
}

void BackendConsumer::HandleTagDeclRequiredDefinition(const TagDecl *D) {
  Gen->HandleTagDeclRequiredDefinition(D);
}

void BackendConsumer::CompleteTentativeDefinition(VarDecl *D) {
  Gen->CompleteTentativeDefinition(D);
}

void BackendConsumer::CompleteExternalDeclaration(DeclaratorDecl *D) {
  Gen->CompleteExternalDeclaration(D);
}

void BackendConsumer::AssignInheritanceModel(CXXRecordDecl *RD) {
  Gen->AssignInheritanceModel(RD);
}

void BackendConsumer::HandleVTable(CXXRecordDecl *RD) {
  Gen->HandleVTable(RD);
}

void BackendConsumer::anchor() { }

} // namespace clang

std::unique_ptr<llvm::DiagnosticHandler>
BackendConsumer::createDiagnosticHandler() {
  return DiagConsumer.createDiagnosticHandler();
}

CodeGenAction::CodeGenAction(unsigned _Act, LLVMContext *_VMContext)
    : Act(_Act), VMContext(_VMContext ? _VMContext : new LLVMContext),
      OwnsVMContext(!_VMContext) {}

CodeGenAction::~CodeGenAction() {
  TheModule.reset();
  if (OwnsVMContext)
    delete VMContext;
}

bool CodeGenAction::hasIRSupport() const { return true; }

void CodeGenAction::EndSourceFileAction() {
  ASTFrontendAction::EndSourceFileAction();

  // If the consumer creation failed, do nothing.
  if (!getCompilerInstance().hasASTConsumer())
    return;

  // Steal the module from the consumer.
  TheModule = BEConsumer->takeModule();
}

std::unique_ptr<llvm::Module> CodeGenAction::takeModule() {
  return std::move(TheModule);
}

llvm::LLVMContext *CodeGenAction::takeLLVMContext() {
  OwnsVMContext = false;
  return VMContext;
}

CodeGenerator *CodeGenAction::getCodeGenerator() const {
  return BEConsumer->getCodeGenerator();
}

bool CodeGenAction::BeginSourceFileAction(CompilerInstance &CI) {
  if (CI.getFrontendOpts().GenReducedBMI)
    CI.getLangOpts().setCompilingModule(LangOptions::CMK_ModuleInterface);
  return ASTFrontendAction::BeginSourceFileAction(CI);
}

static std::unique_ptr<raw_pwrite_stream>
GetOutputStream(CompilerInstance &CI, StringRef InFile, BackendAction Action) {
  switch (Action) {
  case Backend_EmitAssembly:
    return CI.createDefaultOutputFile(false, InFile, "s");
  case Backend_EmitLL:
    return CI.createDefaultOutputFile(false, InFile, "ll");
  case Backend_EmitBC:
    return CI.createDefaultOutputFile(true, InFile, "bc");
  case Backend_EmitNothing:
    return nullptr;
  case Backend_EmitMCNull:
    return CI.createNullOutputFile();
  case Backend_EmitObj:
    return CI.createDefaultOutputFile(true, InFile, "o");
  }

  llvm_unreachable("Invalid action!");
}

std::unique_ptr<ASTConsumer>
CodeGenAction::CreateASTConsumer(CompilerInstance &CI, StringRef InFile) {
  BackendAction BA = static_cast<BackendAction>(Act);
  std::unique_ptr<raw_pwrite_stream> OS = CI.takeOutputStream();
  if (!OS)
    OS = GetOutputStream(CI, InFile, BA);

  if (BA != Backend_EmitNothing && !OS)
    return nullptr;

  // Load bitcode modules to link with, if we need to.
  if (clang::loadLinkModules(CI, *VMContext, LinkModules))
    return nullptr;

  CoverageSourceInfo *CoverageInfo = nullptr;
  // Add the preprocessor callback only when the coverage mapping is generated.
  if (CI.getCodeGenOpts().CoverageMapping)
    CoverageInfo = CodeGen::CoverageMappingModuleGen::setUpCoverageCallbacks(
        CI.getPreprocessor());

  std::unique_ptr<BackendConsumer> Result(new BackendConsumer(
      CI, BA, CI.getVirtualFileSystemPtr(), *VMContext, std::move(LinkModules),
      InFile, std::move(OS), CoverageInfo));
  BEConsumer = Result.get();

  // Enable generating macro debug info only when debug info is not disabled and
  // also macro debug info is enabled.
  if (CI.getCodeGenOpts().getDebugInfo() != codegenoptions::NoDebugInfo &&
      CI.getCodeGenOpts().MacroDebugInfo) {
    std::unique_ptr<PPCallbacks> Callbacks =
        std::make_unique<MacroPPCallbacks>(BEConsumer->getCodeGenerator(),
                                            CI.getPreprocessor());
    CI.getPreprocessor().addPPCallbacks(std::move(Callbacks));
  }

  if (CI.getFrontendOpts().GenReducedBMI &&
      !CI.getFrontendOpts().ModuleOutputPath.empty()) {
    std::vector<std::unique_ptr<ASTConsumer>> Consumers(2);
    Consumers[0] = std::make_unique<ReducedBMIGenerator>(
        CI.getPreprocessor(), CI.getModuleCache(),
        CI.getFrontendOpts().ModuleOutputPath, CI.getCodeGenOpts());
    Consumers[1] = std::move(Result);
    return std::make_unique<MultiplexConsumer>(std::move(Consumers));
  }

  return std::move(Result);
}

std::unique_ptr<llvm::Module>
CodeGenAction::loadModule(MemoryBufferRef MBRef) {
  CompilerInstance &CI = getCompilerInstance();
  SourceManager &SM = CI.getSourceManager();

  auto DiagErrors = [&](Error E) -> std::unique_ptr<llvm::Module> {
    unsigned DiagID =
        CI.getDiagnostics().getCustomDiagID(DiagnosticsEngine::Error, "%0");
    handleAllErrors(std::move(E), [&](ErrorInfoBase &EIB) {
      CI.getDiagnostics().Report(DiagID) << EIB.message();
    });
    return {};
  };

  // For ThinLTO backend invocations, ensure that the context
  // merges types based on ODR identifiers. We also need to read
  // the correct module out of a multi-module bitcode file.
  if (!CI.getCodeGenOpts().ThinLTOIndexFile.empty()) {
    VMContext->enableDebugTypeODRUniquing();

    Expected<std::vector<BitcodeModule>> BMsOrErr = getBitcodeModuleList(MBRef);
    if (!BMsOrErr)
      return DiagErrors(BMsOrErr.takeError());
    BitcodeModule *Bm = llvm::lto::findThinLTOModule(*BMsOrErr);
    // We have nothing to do if the file contains no ThinLTO module. This is
    // possible if ThinLTO compilation was not able to split module. Content of
    // the file was already processed by indexing and will be passed to the
    // linker using merged object file.
    if (!Bm) {
      auto M = std::make_unique<llvm::Module>("empty", *VMContext);
      M->setTargetTriple(Triple(CI.getTargetOpts().Triple));
      return M;
    }
    Expected<std::unique_ptr<llvm::Module>> MOrErr =
        Bm->parseModule(*VMContext);
    if (!MOrErr)
      return DiagErrors(MOrErr.takeError());
    return std::move(*MOrErr);
  }

  // Load bitcode modules to link with, if we need to.
  if (clang::loadLinkModules(CI, *VMContext, LinkModules))
    return nullptr;

  // Handle textual IR and bitcode file with one single module.
  llvm::SMDiagnostic Err;
  if (std::unique_ptr<llvm::Module> M = parseIR(MBRef, Err, *VMContext)) {
    // For LLVM IR files, always verify the input and report the error in a way
    // that does not ask people to report an issue for it.
    std::string VerifierErr;
    raw_string_ostream VerifierErrStream(VerifierErr);
    if (llvm::verifyModule(*M, &VerifierErrStream)) {
      CI.getDiagnostics().Report(diag::err_invalid_llvm_ir) << VerifierErr;
      return {};
    }
    return M;
  }

  // If MBRef is a bitcode with multiple modules (e.g., -fsplit-lto-unit
  // output), place the extra modules (actually only one, a regular LTO module)
  // into LinkModules as if we are using -mlink-bitcode-file.
  Expected<std::vector<BitcodeModule>> BMsOrErr = getBitcodeModuleList(MBRef);
  if (BMsOrErr && BMsOrErr->size()) {
    std::unique_ptr<llvm::Module> FirstM;
    for (auto &BM : *BMsOrErr) {
      Expected<std::unique_ptr<llvm::Module>> MOrErr =
          BM.parseModule(*VMContext);
      if (!MOrErr)
        return DiagErrors(MOrErr.takeError());
      if (FirstM)
        LinkModules.push_back({std::move(*MOrErr), /*PropagateAttrs=*/false,
                               /*Internalize=*/false, /*LinkFlags=*/{}});
      else
        FirstM = std::move(*MOrErr);
    }
    if (FirstM)
      return FirstM;
  }
  // If BMsOrErr fails, consume the error and use the error message from
  // parseIR.
  consumeError(BMsOrErr.takeError());

  // Translate from the diagnostic info to the SourceManager location if
  // available.
  // TODO: Unify this with ConvertBackendLocation()
  SourceLocation Loc;
  if (Err.getLineNo() > 0) {
    assert(Err.getColumnNo() >= 0);
    Loc = SM.translateFileLineCol(SM.getFileEntryForID(SM.getMainFileID()),
                                  Err.getLineNo(), Err.getColumnNo() + 1);
  }

  // Strip off a leading diagnostic code if there is one.
  StringRef Msg = Err.getMessage();
  Msg.consume_front("error: ");

  unsigned DiagID =
      CI.getDiagnostics().getCustomDiagID(DiagnosticsEngine::Error, "%0");

  CI.getDiagnostics().Report(Loc, DiagID) << Msg;
  return {};
}

void CodeGenAction::ExecuteAction() {
  if (getCurrentFileKind().getLanguage() != Language::LLVM_IR) {
    this->ASTFrontendAction::ExecuteAction();
    return;
  }

  // If this is an IR file, we have to treat it specially.
  BackendAction BA = static_cast<BackendAction>(Act);
  CompilerInstance &CI = getCompilerInstance();
  auto &CodeGenOpts = CI.getCodeGenOpts();
  auto &Diagnostics = CI.getDiagnostics();
  std::unique_ptr<raw_pwrite_stream> OS =
      GetOutputStream(CI, getCurrentFileOrBufferName(), BA);
  if (BA != Backend_EmitNothing && !OS)
    return;

  SourceManager &SM = CI.getSourceManager();
  FileID FID = SM.getMainFileID();
  std::optional<MemoryBufferRef> MainFile = SM.getBufferOrNone(FID);
  if (!MainFile)
    return;

  TheModule = loadModule(*MainFile);
  if (!TheModule)
    return;

  const TargetOptions &TargetOpts = CI.getTargetOpts();
  if (TheModule->getTargetTriple().str() != TargetOpts.Triple) {
    Diagnostics.Report(SourceLocation(), diag::warn_fe_override_module)
        << TargetOpts.Triple;
    TheModule->setTargetTriple(Triple(TargetOpts.Triple));
  }

  EmbedObject(TheModule.get(), CodeGenOpts, CI.getVirtualFileSystem(),
              Diagnostics);
  EmbedBitcode(TheModule.get(), CodeGenOpts, *MainFile);

  LLVMContext &Ctx = TheModule->getContext();

  // Restore any diagnostic handler previously set before returning from this
  // function.
  struct RAII {
    LLVMContext &Ctx;
    std::unique_ptr<DiagnosticHandler> PrevHandler = Ctx.getDiagnosticHandler();
    ~RAII() { Ctx.setDiagnosticHandler(std::move(PrevHandler)); }
  } _{Ctx};

  // Set clang diagnostic handler. To do this we need to create a fake
  // BackendConsumer.
  BackendConsumer Result(CI, BA, CI.getVirtualFileSystemPtr(), *VMContext,
                         std::move(LinkModules), "", nullptr, nullptr,
                         TheModule.get());

  // Link in each pending link module.
  if (!CodeGenOpts.LinkBitcodePostopt && Result.LinkInModules(&*TheModule))
    return;

  // PR44896: Force DiscardValueNames as false. DiscardValueNames cannot be
  // true here because the valued names are needed for reading textual IR.
  Ctx.setDiscardValueNames(false);
  Ctx.setDiagnosticHandler(Result.createDiagnosticHandler());

  Ctx.setDefaultTargetCPU(TargetOpts.CPU);
  Ctx.setDefaultTargetFeatures(llvm::join(TargetOpts.Features, ","));

  Expected<LLVMRemarkFileHandle> OptRecordFileOrErr =
      setupLLVMOptimizationRemarks(
          Ctx, CodeGenOpts.OptRecordFile, CodeGenOpts.OptRecordPasses,
          CodeGenOpts.OptRecordFormat, CodeGenOpts.DiagnosticsWithHotness,
          CodeGenOpts.DiagnosticsHotnessThreshold);

  if (Error E = OptRecordFileOrErr.takeError()) {
    reportOptRecordError(std::move(E), Diagnostics, CodeGenOpts);
    return;
  }
  LLVMRemarkFileHandle OptRecordFile = std::move(*OptRecordFileOrErr);

  emitBackendOutput(CI, CI.getCodeGenOpts(), TheModule.get(), BA,
                    CI.getFileManager().getVirtualFileSystemPtr(),
                    std::move(OS));
  if (OptRecordFile)
    OptRecordFile->keep();
}

//

void EmitAssemblyAction::anchor() { }
EmitAssemblyAction::EmitAssemblyAction(llvm::LLVMContext *_VMContext)
  : CodeGenAction(Backend_EmitAssembly, _VMContext) {}

void EmitBCAction::anchor() { }
EmitBCAction::EmitBCAction(llvm::LLVMContext *_VMContext)
  : CodeGenAction(Backend_EmitBC, _VMContext) {}

void EmitLLVMAction::anchor() { }
EmitLLVMAction::EmitLLVMAction(llvm::LLVMContext *_VMContext)
  : CodeGenAction(Backend_EmitLL, _VMContext) {}

void EmitLLVMOnlyAction::anchor() { }
EmitLLVMOnlyAction::EmitLLVMOnlyAction(llvm::LLVMContext *_VMContext)
  : CodeGenAction(Backend_EmitNothing, _VMContext) {}

void EmitCodeGenOnlyAction::anchor() { }
EmitCodeGenOnlyAction::EmitCodeGenOnlyAction(llvm::LLVMContext *_VMContext)
  : CodeGenAction(Backend_EmitMCNull, _VMContext) {}

void EmitObjAction::anchor() { }
EmitObjAction::EmitObjAction(llvm::LLVMContext *_VMContext)
  : CodeGenAction(Backend_EmitObj, _VMContext) {}
