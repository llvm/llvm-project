//===--- CodeGenAction.cpp - LLVM Code Generation Frontend Action ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CodeGen/CodeGenAction.h"
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
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/ADT/Hashing.h"
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
#include "llvm/IRReader/IRReader.h"
#include "llvm/LTO/LTOBackend.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Pass.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TimeProfiler.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Transforms/IPO/Internalize.h"

#include <memory>
using namespace clang;
using namespace llvm;

#define DEBUG_TYPE "codegenaction"

namespace clang {
  class BackendConsumer;
  class ClangDiagnosticHandler final : public DiagnosticHandler {
  public:
    ClangDiagnosticHandler(const CodeGenOptions &CGOpts, BackendConsumer *BCon)
        : CodeGenOpts(CGOpts), BackendCon(BCon) {}

    bool handleDiagnostics(const DiagnosticInfo &DI) override;

    bool isAnalysisRemarkEnabled(StringRef PassName) const override {
      return CodeGenOpts.OptimizationRemarkAnalysis.patternMatches(PassName);
    }
    bool isMissedOptRemarkEnabled(StringRef PassName) const override {
      return CodeGenOpts.OptimizationRemarkMissed.patternMatches(PassName);
    }
    bool isPassedOptRemarkEnabled(StringRef PassName) const override {
      return CodeGenOpts.OptimizationRemark.patternMatches(PassName);
    }

    bool isAnyRemarkEnabled() const override {
      return CodeGenOpts.OptimizationRemarkAnalysis.hasValidPattern() ||
             CodeGenOpts.OptimizationRemarkMissed.hasValidPattern() ||
             CodeGenOpts.OptimizationRemark.hasValidPattern();
    }

  private:
    const CodeGenOptions &CodeGenOpts;
    BackendConsumer *BackendCon;
  };

  static void reportOptRecordError(Error E, DiagnosticsEngine &Diags,
                                   const CodeGenOptions CodeGenOpts) {
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

  class BackendConsumer : public ASTConsumer {
    using LinkModule = CodeGenAction::LinkModule;

    virtual void anchor();
    DiagnosticsEngine &Diags;
    BackendAction Action;
    const HeaderSearchOptions &HeaderSearchOpts;
    const CodeGenOptions &CodeGenOpts;
    const TargetOptions &TargetOpts;
    const LangOptions &LangOpts;
    std::unique_ptr<raw_pwrite_stream> AsmOutStream;
    ASTContext *Context;

    Timer LLVMIRGeneration;
    unsigned LLVMIRGenerationRefCount;

    /// True if we've finished generating IR. This prevents us from generating
    /// additional LLVM IR after emitting output in HandleTranslationUnit. This
    /// can happen when Clang plugins trigger additional AST deserialization.
    bool IRGenFinished = false;

    bool TimerIsEnabled = false;

    std::unique_ptr<CodeGenerator> Gen;

    SmallVector<LinkModule, 4> LinkModules;

    // A map from mangled names to their function's source location, used for
    // backend diagnostics as the Clang AST may be unavailable. We actually use
    // the mangled name's hash as the key because mangled names can be very
    // long and take up lots of space. Using a hash can cause name collision,
    // but that is rare and the consequences are pointing to a wrong source
    // location which is not severe. This is a vector instead of an actual map
    // because we optimize for time building this map rather than time
    // retrieving an entry, as backend diagnostics are uncommon.
    std::vector<std::pair<llvm::hash_code, FullSourceLoc>>
        ManglingFullSourceLocs;

    // This is here so that the diagnostic printer knows the module a diagnostic
    // refers to.
    llvm::Module *CurLinkModule = nullptr;

  public:
    BackendConsumer(BackendAction Action, DiagnosticsEngine &Diags,
                    const HeaderSearchOptions &HeaderSearchOpts,
                    const PreprocessorOptions &PPOpts,
                    const CodeGenOptions &CodeGenOpts,
                    const TargetOptions &TargetOpts,
                    const LangOptions &LangOpts, const std::string &InFile,
                    SmallVector<LinkModule, 4> LinkModules,
                    std::unique_ptr<raw_pwrite_stream> OS, LLVMContext &C,
                    CoverageSourceInfo *CoverageInfo = nullptr)
        : Diags(Diags), Action(Action), HeaderSearchOpts(HeaderSearchOpts),
          CodeGenOpts(CodeGenOpts), TargetOpts(TargetOpts), LangOpts(LangOpts),
          AsmOutStream(std::move(OS)), Context(nullptr),
          LLVMIRGeneration("irgen", "LLVM IR Generation Time"),
          LLVMIRGenerationRefCount(0),
          Gen(CreateLLVMCodeGen(Diags, InFile, HeaderSearchOpts, PPOpts,
                                CodeGenOpts, C, CoverageInfo)),
          LinkModules(std::move(LinkModules)) {
      TimerIsEnabled = CodeGenOpts.TimePasses;
      llvm::TimePassesIsEnabled = CodeGenOpts.TimePasses;
      llvm::TimePassesPerRun = CodeGenOpts.TimePassesPerRun;
    }

    // This constructor is used in installing an empty BackendConsumer
    // to use the clang diagnostic handler for IR input files. It avoids
    // initializing the OS field.
    BackendConsumer(BackendAction Action, DiagnosticsEngine &Diags,
                    const HeaderSearchOptions &HeaderSearchOpts,
                    const PreprocessorOptions &PPOpts,
                    const CodeGenOptions &CodeGenOpts,
                    const TargetOptions &TargetOpts,
                    const LangOptions &LangOpts, llvm::Module *Module,
                    SmallVector<LinkModule, 4> LinkModules, LLVMContext &C,
                    CoverageSourceInfo *CoverageInfo = nullptr)
        : Diags(Diags), Action(Action), HeaderSearchOpts(HeaderSearchOpts),
          CodeGenOpts(CodeGenOpts), TargetOpts(TargetOpts), LangOpts(LangOpts),
          Context(nullptr),
          LLVMIRGeneration("irgen", "LLVM IR Generation Time"),
          LLVMIRGenerationRefCount(0),
          Gen(CreateLLVMCodeGen(Diags, "", HeaderSearchOpts, PPOpts,
                                CodeGenOpts, C, CoverageInfo)),
          LinkModules(std::move(LinkModules)), CurLinkModule(Module) {
      TimerIsEnabled = CodeGenOpts.TimePasses;
      llvm::TimePassesIsEnabled = CodeGenOpts.TimePasses;
      llvm::TimePassesPerRun = CodeGenOpts.TimePassesPerRun;
    }
    llvm::Module *getModule() const { return Gen->GetModule(); }
    std::unique_ptr<llvm::Module> takeModule() {
      return std::unique_ptr<llvm::Module>(Gen->ReleaseModule());
    }

    CodeGenerator *getCodeGenerator() { return Gen.get(); }

    void HandleCXXStaticMemberVarInstantiation(VarDecl *VD) override {
      Gen->HandleCXXStaticMemberVarInstantiation(VD);
    }

    void Initialize(ASTContext &Ctx) override {
      assert(!Context && "initialized multiple times");

      Context = &Ctx;

      if (TimerIsEnabled)
        LLVMIRGeneration.startTimer();

      Gen->Initialize(Ctx);

      if (TimerIsEnabled)
        LLVMIRGeneration.stopTimer();
    }

    bool HandleTopLevelDecl(DeclGroupRef D) override {
      PrettyStackTraceDecl CrashInfo(*D.begin(), SourceLocation(),
                                     Context->getSourceManager(),
                                     "LLVM IR generation of declaration");

      // Recurse.
      if (TimerIsEnabled) {
        LLVMIRGenerationRefCount += 1;
        if (LLVMIRGenerationRefCount == 1)
          LLVMIRGeneration.startTimer();
      }

      Gen->HandleTopLevelDecl(D);

      if (TimerIsEnabled) {
        LLVMIRGenerationRefCount -= 1;
        if (LLVMIRGenerationRefCount == 0)
          LLVMIRGeneration.stopTimer();
      }

      return true;
    }

    void HandleInlineFunctionDefinition(FunctionDecl *D) override {
      PrettyStackTraceDecl CrashInfo(D, SourceLocation(),
                                     Context->getSourceManager(),
                                     "LLVM IR generation of inline function");
      if (TimerIsEnabled)
        LLVMIRGeneration.startTimer();

      Gen->HandleInlineFunctionDefinition(D);

      if (TimerIsEnabled)
        LLVMIRGeneration.stopTimer();
    }

    void HandleInterestingDecl(DeclGroupRef D) override {
      // Ignore interesting decls from the AST reader after IRGen is finished.
      if (!IRGenFinished)
        HandleTopLevelDecl(D);
    }

    // Links each entry in LinkModules into our module.  Returns true on error.
    bool LinkInModules() {
      for (auto &LM : LinkModules) {
        if (LM.PropagateAttrs)
          for (Function &F : *LM.Module) {
            // Skip intrinsics. Keep consistent with how intrinsics are created
            // in LLVM IR.
            if (F.isIntrinsic())
              continue;
            Gen->CGM().addDefaultFunctionDefinitionAttributes(F);
          }

        CurLinkModule = LM.Module.get();

        bool Err;
        if (LM.Internalize) {
          Err = Linker::linkModules(
              *getModule(), std::move(LM.Module), LM.LinkFlags,
              [](llvm::Module &M, const llvm::StringSet<> &GVS) {
                internalizeModule(M, [&GVS](const llvm::GlobalValue &GV) {
                  return !GV.hasName() || (GVS.count(GV.getName()) == 0);
                });
              });
        } else {
          Err = Linker::linkModules(*getModule(), std::move(LM.Module),
                                    LM.LinkFlags);
        }

        if (Err)
          return true;
      }
      return false; // success
    }

    void HandleTranslationUnit(ASTContext &C) override {
      {
        llvm::TimeTraceScope TimeScope("Frontend");
        PrettyStackTraceString CrashInfo("Per-file LLVM IR generation");
        if (TimerIsEnabled) {
          LLVMIRGenerationRefCount += 1;
          if (LLVMIRGenerationRefCount == 1)
            LLVMIRGeneration.startTimer();
        }

        Gen->HandleTranslationUnit(C);

        if (TimerIsEnabled) {
          LLVMIRGenerationRefCount -= 1;
          if (LLVMIRGenerationRefCount == 0)
            LLVMIRGeneration.stopTimer();
        }

        IRGenFinished = true;
      }

      // Silently ignore if we weren't initialized for some reason.
      if (!getModule())
        return;

      LLVMContext &Ctx = getModule()->getContext();
      std::unique_ptr<DiagnosticHandler> OldDiagnosticHandler =
          Ctx.getDiagnosticHandler();
      Ctx.setDiagnosticHandler(std::make_unique<ClangDiagnosticHandler>(
        CodeGenOpts, this));

      Expected<std::unique_ptr<llvm::ToolOutputFile>> OptRecordFileOrErr =
          setupLLVMOptimizationRemarks(
              Ctx, CodeGenOpts.OptRecordFile, CodeGenOpts.OptRecordPasses,
              CodeGenOpts.OptRecordFormat, CodeGenOpts.DiagnosticsWithHotness,
              CodeGenOpts.DiagnosticsHotnessThreshold);

      if (Error E = OptRecordFileOrErr.takeError()) {
        reportOptRecordError(std::move(E), Diags, CodeGenOpts);
        return;
      }

      std::unique_ptr<llvm::ToolOutputFile> OptRecordFile =
          std::move(*OptRecordFileOrErr);

      if (OptRecordFile &&
          CodeGenOpts.getProfileUse() != CodeGenOptions::ProfileNone)
        Ctx.setDiagnosticsHotnessRequested(true);

      // Link each LinkModule into our module.
      if (LinkInModules())
        return;

      for (auto &F : getModule()->functions()) {
        if (const Decl *FD = Gen->GetDeclForMangledName(F.getName())) {
          auto Loc = FD->getASTContext().getFullLoc(FD->getLocation());
          // TODO: use a fast content hash when available.
          auto NameHash = llvm::hash_value(F.getName());
          ManglingFullSourceLocs.push_back(std::make_pair(NameHash, Loc));
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

      EmitBackendOutput(Diags, HeaderSearchOpts, CodeGenOpts, TargetOpts,
                        LangOpts, C.getTargetInfo().getDataLayoutString(),
                        getModule(), Action, std::move(AsmOutStream));

      Ctx.setDiagnosticHandler(std::move(OldDiagnosticHandler));

      if (OptRecordFile)
        OptRecordFile->keep();
    }

    void HandleTagDeclDefinition(TagDecl *D) override {
      PrettyStackTraceDecl CrashInfo(D, SourceLocation(),
                                     Context->getSourceManager(),
                                     "LLVM IR generation of declaration");
      Gen->HandleTagDeclDefinition(D);
    }

    void HandleTagDeclRequiredDefinition(const TagDecl *D) override {
      Gen->HandleTagDeclRequiredDefinition(D);
    }

    void CompleteTentativeDefinition(VarDecl *D) override {
      Gen->CompleteTentativeDefinition(D);
    }

    void CompleteExternalDeclaration(VarDecl *D) override {
      Gen->CompleteExternalDeclaration(D);
    }

    void AssignInheritanceModel(CXXRecordDecl *RD) override {
      Gen->AssignInheritanceModel(RD);
    }

    void HandleVTable(CXXRecordDecl *RD) override {
      Gen->HandleVTable(RD);
    }

    /// Get the best possible source location to represent a diagnostic that
    /// may have associated debug info.
    const FullSourceLoc
    getBestLocationFromDebugLoc(const llvm::DiagnosticInfoWithLocationBase &D,
                                bool &BadDebugInfo, StringRef &Filename,
                                unsigned &Line, unsigned &Column) const;

    Optional<FullSourceLoc> getFunctionSourceLocation(const Function &F) const;

    void DiagnosticHandlerImpl(const llvm::DiagnosticInfo &DI);
    /// Specialized handler for InlineAsm diagnostic.
    /// \return True if the diagnostic has been successfully reported, false
    /// otherwise.
    bool InlineAsmDiagHandler(const llvm::DiagnosticInfoInlineAsm &D);
    /// Specialized handler for diagnostics reported using SMDiagnostic.
    void SrcMgrDiagHandler(const llvm::DiagnosticInfoSrcMgr &D);
    /// Specialized handler for StackSize diagnostic.
    /// \return True if the diagnostic has been successfully reported, false
    /// otherwise.
    bool StackSizeDiagHandler(const llvm::DiagnosticInfoStackSize &D);
    /// Specialized handler for unsupported backend feature diagnostic.
    void UnsupportedDiagHandler(const llvm::DiagnosticInfoUnsupported &D);
    /// Specialized handlers for optimization remarks.
    /// Note that these handlers only accept remarks and they always handle
    /// them.
    void EmitOptimizationMessage(const llvm::DiagnosticInfoOptimizationBase &D,
                                 unsigned DiagID);
    void
    OptimizationRemarkHandler(const llvm::DiagnosticInfoOptimizationBase &D);
    void OptimizationRemarkHandler(
        const llvm::OptimizationRemarkAnalysisFPCommute &D);
    void OptimizationRemarkHandler(
        const llvm::OptimizationRemarkAnalysisAliasing &D);
    void OptimizationFailureHandler(
        const llvm::DiagnosticInfoOptimizationFailure &D);
    void DontCallDiagHandler(const DiagnosticInfoDontCall &D);
  };

  void BackendConsumer::anchor() {}
}

bool ClangDiagnosticHandler::handleDiagnostics(const DiagnosticInfo &DI) {
  BackendCon->DiagnosticHandlerImpl(DI);
  return true;
}

/// ConvertBackendLocation - Convert a location in a temporary llvm::SourceMgr
/// buffer to be a valid FullSourceLoc.
static FullSourceLoc ConvertBackendLocation(const llvm::SMDiagnostic &D,
                                            SourceManager &CSM) {
  // Get both the clang and llvm source managers.  The location is relative to
  // a memory buffer that the LLVM Source Manager is handling, we need to add
  // a copy to the Clang source manager.
  const llvm::SourceMgr &LSM = *D.getSourceMgr();

  // We need to copy the underlying LLVM memory buffer because llvm::SourceMgr
  // already owns its one and clang::SourceManager wants to own its one.
  const MemoryBuffer *LBuf =
  LSM.getMemoryBuffer(LSM.FindBufferContainingLoc(D.getLoc()));

  // Create the copy and transfer ownership to clang::SourceManager.
  // TODO: Avoid copying files into memory.
  std::unique_ptr<llvm::MemoryBuffer> CBuf =
      llvm::MemoryBuffer::getMemBufferCopy(LBuf->getBuffer(),
                                           LBuf->getBufferIdentifier());
  // FIXME: Keep a file ID map instead of creating new IDs for each location.
  FileID FID = CSM.createFileID(std::move(CBuf));

  // Translate the offset into the file.
  unsigned Offset = D.getLoc().getPointer() - LBuf->getBufferStart();
  SourceLocation NewLoc =
  CSM.getLocForStartOfFile(FID).getLocWithOffset(Offset);
  return FullSourceLoc(NewLoc, CSM);
}

#define ComputeDiagID(Severity, GroupName, DiagID)                             \
  do {                                                                         \
    switch (Severity) {                                                        \
    case llvm::DS_Error:                                                       \
      DiagID = diag::err_fe_##GroupName;                                       \
      break;                                                                   \
    case llvm::DS_Warning:                                                     \
      DiagID = diag::warn_fe_##GroupName;                                      \
      break;                                                                   \
    case llvm::DS_Remark:                                                      \
      llvm_unreachable("'remark' severity not expected");                      \
      break;                                                                   \
    case llvm::DS_Note:                                                        \
      DiagID = diag::note_fe_##GroupName;                                      \
      break;                                                                   \
    }                                                                          \
  } while (false)

#define ComputeDiagRemarkID(Severity, GroupName, DiagID)                       \
  do {                                                                         \
    switch (Severity) {                                                        \
    case llvm::DS_Error:                                                       \
      DiagID = diag::err_fe_##GroupName;                                       \
      break;                                                                   \
    case llvm::DS_Warning:                                                     \
      DiagID = diag::warn_fe_##GroupName;                                      \
      break;                                                                   \
    case llvm::DS_Remark:                                                      \
      DiagID = diag::remark_fe_##GroupName;                                    \
      break;                                                                   \
    case llvm::DS_Note:                                                        \
      DiagID = diag::note_fe_##GroupName;                                      \
      break;                                                                   \
    }                                                                          \
  } while (false)

void BackendConsumer::SrcMgrDiagHandler(const llvm::DiagnosticInfoSrcMgr &DI) {
  const llvm::SMDiagnostic &D = DI.getSMDiag();

  unsigned DiagID;
  if (DI.isInlineAsmDiag())
    ComputeDiagID(DI.getSeverity(), inline_asm, DiagID);
  else
    ComputeDiagID(DI.getSeverity(), source_mgr, DiagID);

  // This is for the empty BackendConsumer that uses the clang diagnostic
  // handler for IR input files.
  if (!Context) {
    D.print(nullptr, llvm::errs());
    Diags.Report(DiagID).AddString("cannot compile inline asm");
    return;
  }

  // There are a couple of different kinds of errors we could get here.
  // First, we re-format the SMDiagnostic in terms of a clang diagnostic.

  // Strip "error: " off the start of the message string.
  StringRef Message = D.getMessage();
  (void)Message.consume_front("error: ");

  // If the SMDiagnostic has an inline asm source location, translate it.
  FullSourceLoc Loc;
  if (D.getLoc() != SMLoc())
    Loc = ConvertBackendLocation(D, Context->getSourceManager());

  // If this problem has clang-level source location information, report the
  // issue in the source with a note showing the instantiated
  // code.
  if (DI.isInlineAsmDiag()) {
    SourceLocation LocCookie =
        SourceLocation::getFromRawEncoding(DI.getLocCookie());
    if (LocCookie.isValid()) {
      Diags.Report(LocCookie, DiagID).AddString(Message);

      if (D.getLoc().isValid()) {
        DiagnosticBuilder B = Diags.Report(Loc, diag::note_fe_inline_asm_here);
        // Convert the SMDiagnostic ranges into SourceRange and attach them
        // to the diagnostic.
        for (const std::pair<unsigned, unsigned> &Range : D.getRanges()) {
          unsigned Column = D.getColumnNo();
          B << SourceRange(Loc.getLocWithOffset(Range.first - Column),
                           Loc.getLocWithOffset(Range.second - Column));
        }
      }
      return;
    }
  }

  // Otherwise, report the backend issue as occurring in the generated .s file.
  // If Loc is invalid, we still need to report the issue, it just gets no
  // location info.
  Diags.Report(Loc, DiagID).AddString(Message);
}

bool
BackendConsumer::InlineAsmDiagHandler(const llvm::DiagnosticInfoInlineAsm &D) {
  unsigned DiagID;
  ComputeDiagID(D.getSeverity(), inline_asm, DiagID);
  std::string Message = D.getMsgStr().str();

  // If this problem has clang-level source location information, report the
  // issue as being a problem in the source with a note showing the instantiated
  // code.
  SourceLocation LocCookie =
      SourceLocation::getFromRawEncoding(D.getLocCookie());
  if (LocCookie.isValid())
    Diags.Report(LocCookie, DiagID).AddString(Message);
  else {
    // Otherwise, report the backend diagnostic as occurring in the generated
    // .s file.
    // If Loc is invalid, we still need to report the diagnostic, it just gets
    // no location info.
    FullSourceLoc Loc;
    Diags.Report(Loc, DiagID).AddString(Message);
  }
  // We handled all the possible severities.
  return true;
}

bool
BackendConsumer::StackSizeDiagHandler(const llvm::DiagnosticInfoStackSize &D) {
  if (D.getSeverity() != llvm::DS_Warning)
    // For now, the only support we have for StackSize diagnostic is warning.
    // We do not know how to format other severities.
    return false;

  auto Loc = getFunctionSourceLocation(D.getFunction());
  if (!Loc)
    return false;

  // FIXME: Shouldn't need to truncate to uint32_t
  Diags.Report(*Loc, diag::warn_fe_frame_larger_than)
      << static_cast<uint32_t>(D.getStackSize())
      << static_cast<uint32_t>(D.getStackLimit())
      << llvm::demangle(D.getFunction().getName().str());
  return true;
}

const FullSourceLoc BackendConsumer::getBestLocationFromDebugLoc(
    const llvm::DiagnosticInfoWithLocationBase &D, bool &BadDebugInfo,
    StringRef &Filename, unsigned &Line, unsigned &Column) const {
  SourceManager &SourceMgr = Context->getSourceManager();
  FileManager &FileMgr = SourceMgr.getFileManager();
  SourceLocation DILoc;

  if (D.isLocationAvailable()) {
    D.getLocation(Filename, Line, Column);
    if (Line > 0) {
      auto FE = FileMgr.getFile(Filename);
      if (!FE)
        FE = FileMgr.getFile(D.getAbsolutePath());
      if (FE) {
        // If -gcolumn-info was not used, Column will be 0. This upsets the
        // source manager, so pass 1 if Column is not set.
        DILoc = SourceMgr.translateFileLineCol(*FE, Line, Column ? Column : 1);
      }
    }
    BadDebugInfo = DILoc.isInvalid();
  }

  // If a location isn't available, try to approximate it using the associated
  // function definition. We use the definition's right brace to differentiate
  // from diagnostics that genuinely relate to the function itself.
  FullSourceLoc Loc(DILoc, SourceMgr);
  if (Loc.isInvalid()) {
    if (auto MaybeLoc = getFunctionSourceLocation(D.getFunction()))
      Loc = *MaybeLoc;
  }

  if (DILoc.isInvalid() && D.isLocationAvailable())
    // If we were not able to translate the file:line:col information
    // back to a SourceLocation, at least emit a note stating that
    // we could not translate this location. This can happen in the
    // case of #line directives.
    Diags.Report(Loc, diag::note_fe_backend_invalid_loc)
        << Filename << Line << Column;

  return Loc;
}

Optional<FullSourceLoc>
BackendConsumer::getFunctionSourceLocation(const Function &F) const {
  auto Hash = llvm::hash_value(F.getName());
  for (const auto &Pair : ManglingFullSourceLocs) {
    if (Pair.first == Hash)
      return Pair.second;
  }
  return Optional<FullSourceLoc>();
}

void BackendConsumer::UnsupportedDiagHandler(
    const llvm::DiagnosticInfoUnsupported &D) {
  // We only support warnings or errors.
  assert(D.getSeverity() == llvm::DS_Error ||
         D.getSeverity() == llvm::DS_Warning);

  StringRef Filename;
  unsigned Line, Column;
  bool BadDebugInfo = false;
  FullSourceLoc Loc;
  std::string Msg;
  raw_string_ostream MsgStream(Msg);

  // Context will be nullptr for IR input files, we will construct the diag
  // message from llvm::DiagnosticInfoUnsupported.
  if (Context != nullptr) {
    Loc = getBestLocationFromDebugLoc(D, BadDebugInfo, Filename, Line, Column);
    MsgStream << D.getMessage();
  } else {
    DiagnosticPrinterRawOStream DP(MsgStream);
    D.print(DP);
  }

  auto DiagType = D.getSeverity() == llvm::DS_Error
                      ? diag::err_fe_backend_unsupported
                      : diag::warn_fe_backend_unsupported;
  Diags.Report(Loc, DiagType) << MsgStream.str();

  if (BadDebugInfo)
    // If we were not able to translate the file:line:col information
    // back to a SourceLocation, at least emit a note stating that
    // we could not translate this location. This can happen in the
    // case of #line directives.
    Diags.Report(Loc, diag::note_fe_backend_invalid_loc)
        << Filename << Line << Column;
}

void BackendConsumer::EmitOptimizationMessage(
    const llvm::DiagnosticInfoOptimizationBase &D, unsigned DiagID) {
  // We only support warnings and remarks.
  assert(D.getSeverity() == llvm::DS_Remark ||
         D.getSeverity() == llvm::DS_Warning);

  StringRef Filename;
  unsigned Line, Column;
  bool BadDebugInfo = false;
  FullSourceLoc Loc;
  std::string Msg;
  raw_string_ostream MsgStream(Msg);

  // Context will be nullptr for IR input files, we will construct the remark
  // message from llvm::DiagnosticInfoOptimizationBase.
  if (Context != nullptr) {
    Loc = getBestLocationFromDebugLoc(D, BadDebugInfo, Filename, Line, Column);
    MsgStream << D.getMsg();
  } else {
    DiagnosticPrinterRawOStream DP(MsgStream);
    D.print(DP);
  }

  if (D.getHotness())
    MsgStream << " (hotness: " << *D.getHotness() << ")";

  Diags.Report(Loc, DiagID)
      << AddFlagValue(D.getPassName())
      << MsgStream.str();

  if (BadDebugInfo)
    // If we were not able to translate the file:line:col information
    // back to a SourceLocation, at least emit a note stating that
    // we could not translate this location. This can happen in the
    // case of #line directives.
    Diags.Report(Loc, diag::note_fe_backend_invalid_loc)
        << Filename << Line << Column;
}

void BackendConsumer::OptimizationRemarkHandler(
    const llvm::DiagnosticInfoOptimizationBase &D) {
  // Without hotness information, don't show noisy remarks.
  if (D.isVerbose() && !D.getHotness())
    return;

  if (D.isPassed()) {
    // Optimization remarks are active only if the -Rpass flag has a regular
    // expression that matches the name of the pass name in \p D.
    if (CodeGenOpts.OptimizationRemark.patternMatches(D.getPassName()))
      EmitOptimizationMessage(D, diag::remark_fe_backend_optimization_remark);
  } else if (D.isMissed()) {
    // Missed optimization remarks are active only if the -Rpass-missed
    // flag has a regular expression that matches the name of the pass
    // name in \p D.
    if (CodeGenOpts.OptimizationRemarkMissed.patternMatches(D.getPassName()))
      EmitOptimizationMessage(
          D, diag::remark_fe_backend_optimization_remark_missed);
  } else {
    assert(D.isAnalysis() && "Unknown remark type");

    bool ShouldAlwaysPrint = false;
    if (auto *ORA = dyn_cast<llvm::OptimizationRemarkAnalysis>(&D))
      ShouldAlwaysPrint = ORA->shouldAlwaysPrint();

    if (ShouldAlwaysPrint ||
        CodeGenOpts.OptimizationRemarkAnalysis.patternMatches(D.getPassName()))
      EmitOptimizationMessage(
          D, diag::remark_fe_backend_optimization_remark_analysis);
  }
}

void BackendConsumer::OptimizationRemarkHandler(
    const llvm::OptimizationRemarkAnalysisFPCommute &D) {
  // Optimization analysis remarks are active if the pass name is set to
  // llvm::DiagnosticInfo::AlwasyPrint or if the -Rpass-analysis flag has a
  // regular expression that matches the name of the pass name in \p D.

  if (D.shouldAlwaysPrint() ||
      CodeGenOpts.OptimizationRemarkAnalysis.patternMatches(D.getPassName()))
    EmitOptimizationMessage(
        D, diag::remark_fe_backend_optimization_remark_analysis_fpcommute);
}

void BackendConsumer::OptimizationRemarkHandler(
    const llvm::OptimizationRemarkAnalysisAliasing &D) {
  // Optimization analysis remarks are active if the pass name is set to
  // llvm::DiagnosticInfo::AlwasyPrint or if the -Rpass-analysis flag has a
  // regular expression that matches the name of the pass name in \p D.

  if (D.shouldAlwaysPrint() ||
      CodeGenOpts.OptimizationRemarkAnalysis.patternMatches(D.getPassName()))
    EmitOptimizationMessage(
        D, diag::remark_fe_backend_optimization_remark_analysis_aliasing);
}

void BackendConsumer::OptimizationFailureHandler(
    const llvm::DiagnosticInfoOptimizationFailure &D) {
  EmitOptimizationMessage(D, diag::warn_fe_backend_optimization_failure);
}

void BackendConsumer::DontCallDiagHandler(const DiagnosticInfoDontCall &D) {
  SourceLocation LocCookie =
      SourceLocation::getFromRawEncoding(D.getLocCookie());

  // FIXME: we can't yet diagnose indirect calls. When/if we can, we
  // should instead assert that LocCookie.isValid().
  if (!LocCookie.isValid())
    return;

  Diags.Report(LocCookie, D.getSeverity() == DiagnosticSeverity::DS_Error
                              ? diag::err_fe_backend_error_attr
                              : diag::warn_fe_backend_warning_attr)
      << llvm::demangle(D.getFunctionName().str()) << D.getNote();
}

/// This function is invoked when the backend needs
/// to report something to the user.
void BackendConsumer::DiagnosticHandlerImpl(const DiagnosticInfo &DI) {
  unsigned DiagID = diag::err_fe_inline_asm;
  llvm::DiagnosticSeverity Severity = DI.getSeverity();
  // Get the diagnostic ID based.
  switch (DI.getKind()) {
  case llvm::DK_InlineAsm:
    if (InlineAsmDiagHandler(cast<DiagnosticInfoInlineAsm>(DI)))
      return;
    ComputeDiagID(Severity, inline_asm, DiagID);
    break;
  case llvm::DK_SrcMgr:
    SrcMgrDiagHandler(cast<DiagnosticInfoSrcMgr>(DI));
    return;
  case llvm::DK_StackSize:
    if (StackSizeDiagHandler(cast<DiagnosticInfoStackSize>(DI)))
      return;
    ComputeDiagID(Severity, backend_frame_larger_than, DiagID);
    break;
  case DK_Linker:
    ComputeDiagID(Severity, linking_module, DiagID);
    break;
  case llvm::DK_OptimizationRemark:
    // Optimization remarks are always handled completely by this
    // handler. There is no generic way of emitting them.
    OptimizationRemarkHandler(cast<OptimizationRemark>(DI));
    return;
  case llvm::DK_OptimizationRemarkMissed:
    // Optimization remarks are always handled completely by this
    // handler. There is no generic way of emitting them.
    OptimizationRemarkHandler(cast<OptimizationRemarkMissed>(DI));
    return;
  case llvm::DK_OptimizationRemarkAnalysis:
    // Optimization remarks are always handled completely by this
    // handler. There is no generic way of emitting them.
    OptimizationRemarkHandler(cast<OptimizationRemarkAnalysis>(DI));
    return;
  case llvm::DK_OptimizationRemarkAnalysisFPCommute:
    // Optimization remarks are always handled completely by this
    // handler. There is no generic way of emitting them.
    OptimizationRemarkHandler(cast<OptimizationRemarkAnalysisFPCommute>(DI));
    return;
  case llvm::DK_OptimizationRemarkAnalysisAliasing:
    // Optimization remarks are always handled completely by this
    // handler. There is no generic way of emitting them.
    OptimizationRemarkHandler(cast<OptimizationRemarkAnalysisAliasing>(DI));
    return;
  case llvm::DK_MachineOptimizationRemark:
    // Optimization remarks are always handled completely by this
    // handler. There is no generic way of emitting them.
    OptimizationRemarkHandler(cast<MachineOptimizationRemark>(DI));
    return;
  case llvm::DK_MachineOptimizationRemarkMissed:
    // Optimization remarks are always handled completely by this
    // handler. There is no generic way of emitting them.
    OptimizationRemarkHandler(cast<MachineOptimizationRemarkMissed>(DI));
    return;
  case llvm::DK_MachineOptimizationRemarkAnalysis:
    // Optimization remarks are always handled completely by this
    // handler. There is no generic way of emitting them.
    OptimizationRemarkHandler(cast<MachineOptimizationRemarkAnalysis>(DI));
    return;
  case llvm::DK_OptimizationFailure:
    // Optimization failures are always handled completely by this
    // handler.
    OptimizationFailureHandler(cast<DiagnosticInfoOptimizationFailure>(DI));
    return;
  case llvm::DK_Unsupported:
    UnsupportedDiagHandler(cast<DiagnosticInfoUnsupported>(DI));
    return;
  case llvm::DK_DontCall:
    DontCallDiagHandler(cast<DiagnosticInfoDontCall>(DI));
    return;
  default:
    // Plugin IDs are not bound to any value as they are set dynamically.
    ComputeDiagRemarkID(Severity, backend_plugin, DiagID);
    break;
  }
  std::string MsgStorage;
  {
    raw_string_ostream Stream(MsgStorage);
    DiagnosticPrinterRawOStream DP(Stream);
    DI.print(DP);
  }

  if (DI.getKind() == DK_Linker) {
    assert(CurLinkModule && "CurLinkModule must be set for linker diagnostics");
    Diags.Report(DiagID) << CurLinkModule->getModuleIdentifier() << MsgStorage;
    return;
  }

  // Report the backend message using the usual diagnostic mechanism.
  FullSourceLoc Loc;
  Diags.Report(Loc, DiagID).AddString(MsgStorage);
}
#undef ComputeDiagID

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
  if (LinkModules.empty())
    for (const CodeGenOptions::BitcodeFileToLink &F :
         CI.getCodeGenOpts().LinkBitcodeFiles) {
      auto BCBuf = CI.getFileManager().getBufferForFile(F.Filename);
      if (!BCBuf) {
        CI.getDiagnostics().Report(diag::err_cannot_open_file)
            << F.Filename << BCBuf.getError().message();
        LinkModules.clear();
        return nullptr;
      }

      Expected<std::unique_ptr<llvm::Module>> ModuleOrErr =
          getOwningLazyBitcodeModule(std::move(*BCBuf), *VMContext);
      if (!ModuleOrErr) {
        handleAllErrors(ModuleOrErr.takeError(), [&](ErrorInfoBase &EIB) {
          CI.getDiagnostics().Report(diag::err_cannot_open_file)
              << F.Filename << EIB.message();
        });
        LinkModules.clear();
        return nullptr;
      }
      LinkModules.push_back({std::move(ModuleOrErr.get()), F.PropagateAttrs,
                             F.Internalize, F.LinkFlags});
    }

  CoverageSourceInfo *CoverageInfo = nullptr;
  // Add the preprocessor callback only when the coverage mapping is generated.
  if (CI.getCodeGenOpts().CoverageMapping)
    CoverageInfo = CodeGen::CoverageMappingModuleGen::setUpCoverageCallbacks(
        CI.getPreprocessor());

  std::unique_ptr<BackendConsumer> Result(new BackendConsumer(
      BA, CI.getDiagnostics(), CI.getHeaderSearchOpts(),
      CI.getPreprocessorOpts(), CI.getCodeGenOpts(), CI.getTargetOpts(),
      CI.getLangOpts(), std::string(InFile), std::move(LinkModules),
      std::move(OS), *VMContext, CoverageInfo));
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

  return std::move(Result);
}

std::unique_ptr<llvm::Module>
CodeGenAction::loadModule(MemoryBufferRef MBRef) {
  CompilerInstance &CI = getCompilerInstance();
  SourceManager &SM = CI.getSourceManager();

  // For ThinLTO backend invocations, ensure that the context
  // merges types based on ODR identifiers. We also need to read
  // the correct module out of a multi-module bitcode file.
  if (!CI.getCodeGenOpts().ThinLTOIndexFile.empty()) {
    VMContext->enableDebugTypeODRUniquing();

    auto DiagErrors = [&](Error E) -> std::unique_ptr<llvm::Module> {
      unsigned DiagID =
          CI.getDiagnostics().getCustomDiagID(DiagnosticsEngine::Error, "%0");
      handleAllErrors(std::move(E), [&](ErrorInfoBase &EIB) {
        CI.getDiagnostics().Report(DiagID) << EIB.message();
      });
      return {};
    };

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
      M->setTargetTriple(CI.getTargetOpts().Triple);
      return M;
    }
    Expected<std::unique_ptr<llvm::Module>> MOrErr =
        Bm->parseModule(*VMContext);
    if (!MOrErr)
      return DiagErrors(MOrErr.takeError());
    return std::move(*MOrErr);
  }

  llvm::SMDiagnostic Err;
  if (std::unique_ptr<llvm::Module> M = parseIR(MBRef, Err, *VMContext))
    return M;

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
  if (Msg.startswith("error: "))
    Msg = Msg.substr(7);

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
  Optional<MemoryBufferRef> MainFile = SM.getBufferOrNone(FID);
  if (!MainFile)
    return;

  TheModule = loadModule(*MainFile);
  if (!TheModule)
    return;

  const TargetOptions &TargetOpts = CI.getTargetOpts();
  if (TheModule->getTargetTriple() != TargetOpts.Triple) {
    Diagnostics.Report(SourceLocation(), diag::warn_fe_override_module)
        << TargetOpts.Triple;
    TheModule->setTargetTriple(TargetOpts.Triple);
  }

  EmbedObject(TheModule.get(), CodeGenOpts, Diagnostics);
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
  BackendConsumer Result(BA, CI.getDiagnostics(), CI.getHeaderSearchOpts(),
                         CI.getPreprocessorOpts(), CI.getCodeGenOpts(),
                         CI.getTargetOpts(), CI.getLangOpts(), TheModule.get(),
                         std::move(LinkModules), *VMContext, nullptr);
  // PR44896: Force DiscardValueNames as false. DiscardValueNames cannot be
  // true here because the valued names are needed for reading textual IR.
  Ctx.setDiscardValueNames(false);
  Ctx.setDiagnosticHandler(
      std::make_unique<ClangDiagnosticHandler>(CodeGenOpts, &Result));

  Expected<std::unique_ptr<llvm::ToolOutputFile>> OptRecordFileOrErr =
      setupLLVMOptimizationRemarks(
          Ctx, CodeGenOpts.OptRecordFile, CodeGenOpts.OptRecordPasses,
          CodeGenOpts.OptRecordFormat, CodeGenOpts.DiagnosticsWithHotness,
          CodeGenOpts.DiagnosticsHotnessThreshold);

  if (Error E = OptRecordFileOrErr.takeError()) {
    reportOptRecordError(std::move(E), Diagnostics, CodeGenOpts);
    return;
  }
  std::unique_ptr<llvm::ToolOutputFile> OptRecordFile =
      std::move(*OptRecordFileOrErr);

  EmitBackendOutput(Diagnostics, CI.getHeaderSearchOpts(), CodeGenOpts,
                    TargetOpts, CI.getLangOpts(),
                    CI.getTarget().getDataLayoutString(), TheModule.get(), BA,
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
