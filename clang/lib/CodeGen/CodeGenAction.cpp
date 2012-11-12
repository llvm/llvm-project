//===--- CodeGenAction.cpp - LLVM Code Generation Frontend Action ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/CodeGen/CodeGenAction.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclGroup.h"
#include "clang/CodeGen/BackendUtil.h"
#include "clang/CodeGen/ModuleBuilder.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "llvm/LLVMContext.h"
#include "llvm/Linker.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/Support/IRReader.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/Timer.h"
using namespace clang;
using namespace llvm;

namespace clang {
  class BackendConsumer : public ASTConsumer {
    virtual void anchor();
    DiagnosticsEngine &Diags;
    BackendAction Action;
    const CodeGenOptions &CodeGenOpts;
    const TargetOptions &TargetOpts;
    const LangOptions &LangOpts;
    raw_ostream *AsmOutStream;
    ASTContext *Context;

    Timer LLVMIRGeneration;

    OwningPtr<CodeGenerator> Gen;

    OwningPtr<llvm::Module> TheModule, LinkModule;

  public:
    BackendConsumer(BackendAction action, DiagnosticsEngine &_Diags,
                    const CodeGenOptions &compopts,
                    const TargetOptions &targetopts,
                    const LangOptions &langopts,
                    bool TimePasses,
                    const std::string &infile,
                    llvm::Module *LinkModule,
                    raw_ostream *OS,
                    LLVMContext &C) :
      Diags(_Diags),
      Action(action),
      CodeGenOpts(compopts),
      TargetOpts(targetopts),
      LangOpts(langopts),
      AsmOutStream(OS),
      Context(), 
      LLVMIRGeneration("LLVM IR Generation Time"),
      Gen(CreateLLVMCodeGen(Diags, infile, compopts, C)),
      LinkModule(LinkModule)
    {
      llvm::TimePassesIsEnabled = TimePasses;
    }

    llvm::Module *takeModule() { return TheModule.take(); }
    llvm::Module *takeLinkModule() { return LinkModule.take(); }

    virtual void HandleCXXStaticMemberVarInstantiation(VarDecl *VD) {
      Gen->HandleCXXStaticMemberVarInstantiation(VD);
    }

    virtual void Initialize(ASTContext &Ctx) {
      Context = &Ctx;

      if (llvm::TimePassesIsEnabled)
        LLVMIRGeneration.startTimer();

      Gen->Initialize(Ctx);

      TheModule.reset(Gen->GetModule());

      if (llvm::TimePassesIsEnabled)
        LLVMIRGeneration.stopTimer();
    }

    virtual bool HandleTopLevelDecl(DeclGroupRef D) {
      PrettyStackTraceDecl CrashInfo(*D.begin(), SourceLocation(),
                                     Context->getSourceManager(),
                                     "LLVM IR generation of declaration");

      if (llvm::TimePassesIsEnabled)
        LLVMIRGeneration.startTimer();

      Gen->HandleTopLevelDecl(D);

      if (llvm::TimePassesIsEnabled)
        LLVMIRGeneration.stopTimer();

      return true;
    }

    virtual void HandleTranslationUnit(ASTContext &C) {
      {
        PrettyStackTraceString CrashInfo("Per-file LLVM IR generation");
        if (llvm::TimePassesIsEnabled)
          LLVMIRGeneration.startTimer();

        Gen->HandleTranslationUnit(C);

        if (llvm::TimePassesIsEnabled)
          LLVMIRGeneration.stopTimer();
      }

      // Silently ignore if we weren't initialized for some reason.
      if (!TheModule)
        return;

      // Make sure IR generation is happy with the module. This is released by
      // the module provider.
      llvm::Module *M = Gen->ReleaseModule();
      if (!M) {
        // The module has been released by IR gen on failures, do not double
        // free.
        TheModule.take();
        return;
      }

      assert(TheModule.get() == M &&
             "Unexpected module change during IR generation");

      // Link LinkModule into this module if present, preserving its validity.
      if (LinkModule) {
        std::string ErrorMsg;
        if (Linker::LinkModules(M, LinkModule.get(), Linker::PreserveSource,
                                &ErrorMsg)) {
          Diags.Report(diag::err_fe_cannot_link_module)
            << LinkModule->getModuleIdentifier() << ErrorMsg;
          return;
        }
      }

      // Install an inline asm handler so that diagnostics get printed through
      // our diagnostics hooks.
      LLVMContext &Ctx = TheModule->getContext();
      LLVMContext::InlineAsmDiagHandlerTy OldHandler =
        Ctx.getInlineAsmDiagnosticHandler();
      void *OldContext = Ctx.getInlineAsmDiagnosticContext();
      Ctx.setInlineAsmDiagnosticHandler(InlineAsmDiagHandler, this);

      EmitBackendOutput(Diags, CodeGenOpts, TargetOpts, LangOpts,
                        TheModule.get(), Action, AsmOutStream);
      
      Ctx.setInlineAsmDiagnosticHandler(OldHandler, OldContext);
    }

    virtual void HandleTagDeclDefinition(TagDecl *D) {
      PrettyStackTraceDecl CrashInfo(D, SourceLocation(),
                                     Context->getSourceManager(),
                                     "LLVM IR generation of declaration");
      Gen->HandleTagDeclDefinition(D);
    }

    virtual void CompleteTentativeDefinition(VarDecl *D) {
      Gen->CompleteTentativeDefinition(D);
    }

    virtual void HandleVTable(CXXRecordDecl *RD, bool DefinitionRequired) {
      Gen->HandleVTable(RD, DefinitionRequired);
    }

    static void InlineAsmDiagHandler(const llvm::SMDiagnostic &SM,void *Context,
                                     unsigned LocCookie) {
      SourceLocation Loc = SourceLocation::getFromRawEncoding(LocCookie);
      ((BackendConsumer*)Context)->InlineAsmDiagHandler2(SM, Loc);
    }

    void InlineAsmDiagHandler2(const llvm::SMDiagnostic &,
                               SourceLocation LocCookie);
  };
  
  void BackendConsumer::anchor() {}
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
  llvm::MemoryBuffer *CBuf =
  llvm::MemoryBuffer::getMemBufferCopy(LBuf->getBuffer(),
                                       LBuf->getBufferIdentifier());
  FileID FID = CSM.createFileIDForMemBuffer(CBuf);

  // Translate the offset into the file.
  unsigned Offset = D.getLoc().getPointer()  - LBuf->getBufferStart();
  SourceLocation NewLoc =
  CSM.getLocForStartOfFile(FID).getLocWithOffset(Offset);
  return FullSourceLoc(NewLoc, CSM);
}


/// InlineAsmDiagHandler2 - This function is invoked when the backend hits an
/// error parsing inline asm.  The SMDiagnostic indicates the error relative to
/// the temporary memory buffer that the inline asm parser has set up.
void BackendConsumer::InlineAsmDiagHandler2(const llvm::SMDiagnostic &D,
                                            SourceLocation LocCookie) {
  // There are a couple of different kinds of errors we could get here.  First,
  // we re-format the SMDiagnostic in terms of a clang diagnostic.

  // Strip "error: " off the start of the message string.
  StringRef Message = D.getMessage();
  if (Message.startswith("error: "))
    Message = Message.substr(7);

  // If the SMDiagnostic has an inline asm source location, translate it.
  FullSourceLoc Loc;
  if (D.getLoc() != SMLoc())
    Loc = ConvertBackendLocation(D, Context->getSourceManager());
  

  // If this problem has clang-level source location information, report the
  // issue as being an error in the source with a note showing the instantiated
  // code.
  if (LocCookie.isValid()) {
    Diags.Report(LocCookie, diag::err_fe_inline_asm).AddString(Message);
    
    if (D.getLoc().isValid()) {
      DiagnosticBuilder B = Diags.Report(Loc, diag::note_fe_inline_asm_here);
      // Convert the SMDiagnostic ranges into SourceRange and attach them
      // to the diagnostic.
      for (unsigned i = 0, e = D.getRanges().size(); i != e; ++i) {
        std::pair<unsigned, unsigned> Range = D.getRanges()[i];
        unsigned Column = D.getColumnNo();
        B << SourceRange(Loc.getLocWithOffset(Range.first - Column),
                         Loc.getLocWithOffset(Range.second - Column));
      }
    }
    return;
  }
  
  // Otherwise, report the backend error as occurring in the generated .s file.
  // If Loc is invalid, we still need to report the error, it just gets no
  // location info.
  Diags.Report(Loc, diag::err_fe_inline_asm).AddString(Message);
}

//

CodeGenAction::CodeGenAction(unsigned _Act, LLVMContext *_VMContext)
  : Act(_Act), LinkModule(0),
    VMContext(_VMContext ? _VMContext : new LLVMContext),
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

  // If we were given a link module, release consumer's ownership of it.
  if (LinkModule)
    BEConsumer->takeLinkModule();

  // Steal the module from the consumer.
  TheModule.reset(BEConsumer->takeModule());
}

llvm::Module *CodeGenAction::takeModule() {
  return TheModule.take();
}

llvm::LLVMContext *CodeGenAction::takeLLVMContext() {
  OwnsVMContext = false;
  return VMContext;
}

static raw_ostream *GetOutputStream(CompilerInstance &CI,
                                    StringRef InFile,
                                    BackendAction Action) {
  switch (Action) {
  case Backend_EmitAssembly:
    return CI.createDefaultOutputFile(false, InFile, "s");
  case Backend_EmitLL:
    return CI.createDefaultOutputFile(false, InFile, "ll");
  case Backend_EmitBC:
    return CI.createDefaultOutputFile(true, InFile, "bc");
  case Backend_EmitNothing:
    return 0;
  case Backend_EmitMCNull:
  case Backend_EmitObj:
    return CI.createDefaultOutputFile(true, InFile, "o");
  }

  llvm_unreachable("Invalid action!");
}

ASTConsumer *CodeGenAction::CreateASTConsumer(CompilerInstance &CI,
                                              StringRef InFile) {
  BackendAction BA = static_cast<BackendAction>(Act);
  OwningPtr<raw_ostream> OS(GetOutputStream(CI, InFile, BA));
  if (BA != Backend_EmitNothing && !OS)
    return 0;

  llvm::Module *LinkModuleToUse = LinkModule;

  // If we were not given a link module, and the user requested that one be
  // loaded from bitcode, do so now.
  const std::string &LinkBCFile = CI.getCodeGenOpts().LinkBitcodeFile;
  if (!LinkModuleToUse && !LinkBCFile.empty()) {
    std::string ErrorStr;

    llvm::MemoryBuffer *BCBuf =
      CI.getFileManager().getBufferForFile(LinkBCFile, &ErrorStr);
    if (!BCBuf) {
      CI.getDiagnostics().Report(diag::err_cannot_open_file)
        << LinkBCFile << ErrorStr;
      return 0;
    }

    LinkModuleToUse = getLazyBitcodeModule(BCBuf, *VMContext, &ErrorStr);
    if (!LinkModuleToUse) {
      CI.getDiagnostics().Report(diag::err_cannot_open_file)
        << LinkBCFile << ErrorStr;
      return 0;
    }
  }

  BEConsumer = 
      new BackendConsumer(BA, CI.getDiagnostics(),
                          CI.getCodeGenOpts(), CI.getTargetOpts(),
                          CI.getLangOpts(),
                          CI.getFrontendOpts().ShowTimers, InFile,
                          LinkModuleToUse, OS.take(), *VMContext);
  return BEConsumer;
}

void CodeGenAction::ExecuteAction() {
  // If this is an IR file, we have to treat it specially.
  if (getCurrentFileKind() == IK_LLVM_IR) {
    BackendAction BA = static_cast<BackendAction>(Act);
    CompilerInstance &CI = getCompilerInstance();
    raw_ostream *OS = GetOutputStream(CI, getCurrentFile(), BA);
    if (BA != Backend_EmitNothing && !OS)
      return;

    bool Invalid;
    SourceManager &SM = CI.getSourceManager();
    const llvm::MemoryBuffer *MainFile = SM.getBuffer(SM.getMainFileID(),
                                                      &Invalid);
    if (Invalid)
      return;

    // FIXME: This is stupid, IRReader shouldn't take ownership.
    llvm::MemoryBuffer *MainFileCopy =
      llvm::MemoryBuffer::getMemBufferCopy(MainFile->getBuffer(),
                                           getCurrentFile());

    llvm::SMDiagnostic Err;
    TheModule.reset(ParseIR(MainFileCopy, Err, *VMContext));
    if (!TheModule) {
      // Translate from the diagnostic info to the SourceManager location.
      SourceLocation Loc = SM.translateFileLineCol(
        SM.getFileEntryForID(SM.getMainFileID()), Err.getLineNo(),
        Err.getColumnNo() + 1);

      // Get a custom diagnostic for the error. We strip off a leading
      // diagnostic code if there is one.
      StringRef Msg = Err.getMessage();
      if (Msg.startswith("error: "))
        Msg = Msg.substr(7);

      // Escape '%', which is interpreted as a format character.
      llvm::SmallString<128> EscapedMessage;
      for (unsigned i = 0, e = Msg.size(); i != e; ++i) {
        if (Msg[i] == '%')
          EscapedMessage += '%';
        EscapedMessage += Msg[i];
      }

      unsigned DiagID = CI.getDiagnostics().getCustomDiagID(
          DiagnosticsEngine::Error, EscapedMessage);

      CI.getDiagnostics().Report(Loc, DiagID);
      return;
    }

    EmitBackendOutput(CI.getDiagnostics(), CI.getCodeGenOpts(),
                      CI.getTargetOpts(), CI.getLangOpts(),
                      TheModule.get(),
                      BA, OS);
    return;
  }

  // Otherwise follow the normal AST path.
  this->ASTFrontendAction::ExecuteAction();
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
