//===--- FrontendAction.cpp -----------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/FrontendAction.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclGroup.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Frontend/ChainedIncludesSource.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/Frontend/LayoutOverrideSource.h"
#include "clang/Frontend/MultiplexConsumer.h"
#include "clang/Parse/ParseAST.h"
#include "clang/Serialization/ASTDeserializationListener.h"
#include "clang/Serialization/ASTReader.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/system_error.h"
#include "llvm/Support/Timer.h"
using namespace clang;

namespace {

class DelegatingDeserializationListener : public ASTDeserializationListener {
  ASTDeserializationListener *Previous;

public:
  explicit DelegatingDeserializationListener(
                                           ASTDeserializationListener *Previous)
    : Previous(Previous) { }

  virtual void ReaderInitialized(ASTReader *Reader) {
    if (Previous)
      Previous->ReaderInitialized(Reader);
  }
  virtual void IdentifierRead(serialization::IdentID ID,
                              IdentifierInfo *II) {
    if (Previous)
      Previous->IdentifierRead(ID, II);
  }
  virtual void TypeRead(serialization::TypeIdx Idx, QualType T) {
    if (Previous)
      Previous->TypeRead(Idx, T);
  }
  virtual void DeclRead(serialization::DeclID ID, const Decl *D) {
    if (Previous)
      Previous->DeclRead(ID, D);
  }
  virtual void SelectorRead(serialization::SelectorID ID, Selector Sel) {
    if (Previous)
      Previous->SelectorRead(ID, Sel);
  }
  virtual void MacroDefinitionRead(serialization::PreprocessedEntityID PPID, 
                                   MacroDefinition *MD) {
    if (Previous)
      Previous->MacroDefinitionRead(PPID, MD);
  }
};

/// \brief Dumps deserialized declarations.
class DeserializedDeclsDumper : public DelegatingDeserializationListener {
public:
  explicit DeserializedDeclsDumper(ASTDeserializationListener *Previous)
    : DelegatingDeserializationListener(Previous) { }

  virtual void DeclRead(serialization::DeclID ID, const Decl *D) {
    llvm::outs() << "PCH DECL: " << D->getDeclKindName();
    if (const NamedDecl *ND = dyn_cast<NamedDecl>(D))
      llvm::outs() << " - " << *ND;
    llvm::outs() << "\n";

    DelegatingDeserializationListener::DeclRead(ID, D);
  }
};

/// \brief Checks deserialized declarations and emits error if a name
/// matches one given in command-line using -error-on-deserialized-decl.
class DeserializedDeclsChecker : public DelegatingDeserializationListener {
  ASTContext &Ctx;
  std::set<std::string> NamesToCheck;

public:
  DeserializedDeclsChecker(ASTContext &Ctx,
                           const std::set<std::string> &NamesToCheck,
                           ASTDeserializationListener *Previous)
    : DelegatingDeserializationListener(Previous),
      Ctx(Ctx), NamesToCheck(NamesToCheck) { }

  virtual void DeclRead(serialization::DeclID ID, const Decl *D) {
    if (const NamedDecl *ND = dyn_cast<NamedDecl>(D))
      if (NamesToCheck.find(ND->getNameAsString()) != NamesToCheck.end()) {
        unsigned DiagID
          = Ctx.getDiagnostics().getCustomDiagID(DiagnosticsEngine::Error,
                                                 "%0 was deserialized");
        Ctx.getDiagnostics().Report(Ctx.getFullLoc(D->getLocation()), DiagID)
            << ND->getNameAsString();
      }

    DelegatingDeserializationListener::DeclRead(ID, D);
  }
};

} // end anonymous namespace

FrontendAction::FrontendAction() : Instance(0) {}

FrontendAction::~FrontendAction() {}

void FrontendAction::setCurrentInput(const FrontendInputFile &CurrentInput,
                                     ASTUnit *AST) {
  this->CurrentInput = CurrentInput;
  CurrentASTUnit.reset(AST);
}

ASTConsumer* FrontendAction::CreateWrappedASTConsumer(CompilerInstance &CI,
                                                      StringRef InFile) {
  ASTConsumer* Consumer = CreateASTConsumer(CI, InFile);
  if (!Consumer)
    return 0;

  if (CI.getFrontendOpts().AddPluginActions.size() == 0)
    return Consumer;

  // Make sure the non-plugin consumer is first, so that plugins can't
  // modifiy the AST.
  std::vector<ASTConsumer*> Consumers(1, Consumer);

  for (size_t i = 0, e = CI.getFrontendOpts().AddPluginActions.size();
       i != e; ++i) { 
    // This is O(|plugins| * |add_plugins|), but since both numbers are
    // way below 50 in practice, that's ok.
    for (FrontendPluginRegistry::iterator
        it = FrontendPluginRegistry::begin(),
        ie = FrontendPluginRegistry::end();
        it != ie; ++it) {
      if (it->getName() == CI.getFrontendOpts().AddPluginActions[i]) {
        OwningPtr<PluginASTAction> P(it->instantiate());
        FrontendAction* c = P.get();
        if (P->ParseArgs(CI, CI.getFrontendOpts().AddPluginArgs[i]))
          Consumers.push_back(c->CreateASTConsumer(CI, InFile));
      }
    }
  }

  return new MultiplexConsumer(Consumers);
}


bool FrontendAction::BeginSourceFile(CompilerInstance &CI,
                                     const FrontendInputFile &Input) {
  assert(!Instance && "Already processing a source file!");
  assert(!Input.isEmpty() && "Unexpected empty filename!");
  setCurrentInput(Input);
  setCompilerInstance(&CI);

  StringRef InputFile = Input.getFile();
  bool HasBegunSourceFile = false;
  if (!BeginInvocation(CI))
    goto failure;

  // AST files follow a very different path, since they share objects via the
  // AST unit.
  if (Input.getKind() == IK_AST) {
    assert(!usesPreprocessorOnly() &&
           "Attempt to pass AST file to preprocessor only action!");
    assert(hasASTFileSupport() &&
           "This action does not have AST file support!");

    IntrusiveRefCntPtr<DiagnosticsEngine> Diags(&CI.getDiagnostics());
    std::string Error;
    ASTUnit *AST = ASTUnit::LoadFromASTFile(InputFile, Diags,
                                            CI.getFileSystemOpts());
    if (!AST)
      goto failure;

    setCurrentInput(Input, AST);

    // Set the shared objects, these are reset when we finish processing the
    // file, otherwise the CompilerInstance will happily destroy them.
    CI.setFileManager(&AST->getFileManager());
    CI.setSourceManager(&AST->getSourceManager());
    CI.setPreprocessor(&AST->getPreprocessor());
    CI.setASTContext(&AST->getASTContext());

    // Initialize the action.
    if (!BeginSourceFileAction(CI, InputFile))
      goto failure;

    /// Create the AST consumer.
    CI.setASTConsumer(CreateWrappedASTConsumer(CI, InputFile));
    if (!CI.hasASTConsumer())
      goto failure;

    return true;
  }

  // Set up the file and source managers, if needed.
  if (!CI.hasFileManager())
    CI.createFileManager();
  if (!CI.hasSourceManager())
    CI.createSourceManager(CI.getFileManager());

  // IR files bypass the rest of initialization.
  if (Input.getKind() == IK_LLVM_IR) {
    assert(hasIRSupport() &&
           "This action does not have IR file support!");

    // Inform the diagnostic client we are processing a source file.
    CI.getDiagnosticClient().BeginSourceFile(CI.getLangOpts(), 0);
    HasBegunSourceFile = true;

    // Initialize the action.
    if (!BeginSourceFileAction(CI, InputFile))
      goto failure;

    return true;
  }

  // If the implicit PCH include is actually a directory, rather than
  // a single file, search for a suitable PCH file in that directory.
  if (!CI.getPreprocessorOpts().ImplicitPCHInclude.empty()) {
    FileManager &FileMgr = CI.getFileManager();
    PreprocessorOptions &PPOpts = CI.getPreprocessorOpts();
    StringRef PCHInclude = PPOpts.ImplicitPCHInclude;
    if (const DirectoryEntry *PCHDir = FileMgr.getDirectory(PCHInclude)) {
      llvm::error_code EC;
      SmallString<128> DirNative;
      llvm::sys::path::native(PCHDir->getName(), DirNative);
      bool Found = false;
      for (llvm::sys::fs::directory_iterator Dir(DirNative.str(), EC), DirEnd;
           Dir != DirEnd && !EC; Dir.increment(EC)) {
        // Check whether this is an acceptable AST file.
        if (ASTReader::isAcceptableASTFile(Dir->path(), FileMgr,
                                           CI.getLangOpts(),
                                           CI.getTargetOpts(),
                                           CI.getPreprocessorOpts())) {
          for (unsigned I = 0, N = PPOpts.Includes.size(); I != N; ++I) {
            if (PPOpts.Includes[I] == PPOpts.ImplicitPCHInclude) {
              PPOpts.Includes[I] = Dir->path();
              PPOpts.ImplicitPCHInclude = Dir->path();
              Found = true;
              break;
            }
          }

          assert(Found && "Implicit PCH include not in includes list?");
          break;
        }
      }

      if (!Found) {
        CI.getDiagnostics().Report(diag::err_fe_no_pch_in_dir) << PCHInclude;
        return true;
      }
    }
  }

  // Set up the preprocessor.
  CI.createPreprocessor();

  // Inform the diagnostic client we are processing a source file.
  CI.getDiagnosticClient().BeginSourceFile(CI.getLangOpts(),
                                           &CI.getPreprocessor());
  HasBegunSourceFile = true;

  // Initialize the action.
  if (!BeginSourceFileAction(CI, InputFile))
    goto failure;

  /// Create the AST context and consumer unless this is a preprocessor only
  /// action.
  if (!usesPreprocessorOnly()) {
    CI.createASTContext();

    OwningPtr<ASTConsumer> Consumer(
                                   CreateWrappedASTConsumer(CI, InputFile));
    if (!Consumer)
      goto failure;

    CI.getASTContext().setASTMutationListener(Consumer->GetASTMutationListener());
    CI.getPreprocessor().setPPMutationListener(
      Consumer->GetPPMutationListener());
    
    if (!CI.getPreprocessorOpts().ChainedIncludes.empty()) {
      // Convert headers to PCH and chain them.
      OwningPtr<ExternalASTSource> source;
      source.reset(ChainedIncludesSource::create(CI));
      if (!source)
        goto failure;
      CI.getASTContext().setExternalSource(source);

    } else if (!CI.getPreprocessorOpts().ImplicitPCHInclude.empty()) {
      // Use PCH.
      assert(hasPCHSupport() && "This action does not have PCH support!");
      ASTDeserializationListener *DeserialListener =
          Consumer->GetASTDeserializationListener();
      if (CI.getPreprocessorOpts().DumpDeserializedPCHDecls)
        DeserialListener = new DeserializedDeclsDumper(DeserialListener);
      if (!CI.getPreprocessorOpts().DeserializedPCHDeclsToErrorOn.empty())
        DeserialListener = new DeserializedDeclsChecker(CI.getASTContext(),
                         CI.getPreprocessorOpts().DeserializedPCHDeclsToErrorOn,
                                                        DeserialListener);
      CI.createPCHExternalASTSource(
                                CI.getPreprocessorOpts().ImplicitPCHInclude,
                                CI.getPreprocessorOpts().DisablePCHValidation,
                            CI.getPreprocessorOpts().AllowPCHWithCompilerErrors,
                                DeserialListener);
      if (!CI.getASTContext().getExternalSource())
        goto failure;
    }

    CI.setASTConsumer(Consumer.take());
    if (!CI.hasASTConsumer())
      goto failure;
  }

  // Initialize built-in info as long as we aren't using an external AST
  // source.
  if (!CI.hasASTContext() || !CI.getASTContext().getExternalSource()) {
    Preprocessor &PP = CI.getPreprocessor();
    PP.getBuiltinInfo().InitializeBuiltins(PP.getIdentifierTable(),
                                           PP.getLangOpts());
  }

  // If there is a layout overrides file, attach an external AST source that
  // provides the layouts from that file.
  if (!CI.getFrontendOpts().OverrideRecordLayoutsFile.empty() && 
      CI.hasASTContext() && !CI.getASTContext().getExternalSource()) {
    OwningPtr<ExternalASTSource> 
      Override(new LayoutOverrideSource(
                     CI.getFrontendOpts().OverrideRecordLayoutsFile));
    CI.getASTContext().setExternalSource(Override);
  }
  
  return true;

  // If we failed, reset state since the client will not end up calling the
  // matching EndSourceFile().
  failure:
  if (isCurrentFileAST()) {
    CI.setASTContext(0);
    CI.setPreprocessor(0);
    CI.setSourceManager(0);
    CI.setFileManager(0);
  }

  if (HasBegunSourceFile)
    CI.getDiagnosticClient().EndSourceFile();
  CI.clearOutputFiles(/*EraseFiles=*/true);
  setCurrentInput(FrontendInputFile());
  setCompilerInstance(0);
  return false;
}

bool FrontendAction::Execute() {
  CompilerInstance &CI = getCompilerInstance();

  // Initialize the main file entry. This needs to be delayed until after PCH
  // has loaded.
  if (!isCurrentFileAST()) {
    if (!CI.InitializeSourceManager(getCurrentInput()))
      return false;
  }

  if (CI.hasFrontendTimer()) {
    llvm::TimeRegion Timer(CI.getFrontendTimer());
    ExecuteAction();
  }
  else ExecuteAction();

  return true;
}

void FrontendAction::EndSourceFile() {
  CompilerInstance &CI = getCompilerInstance();

  // Inform the diagnostic client we are done with this source file.
  CI.getDiagnosticClient().EndSourceFile();

  // Finalize the action.
  EndSourceFileAction();

  // Release the consumer and the AST, in that order since the consumer may
  // perform actions in its destructor which require the context.
  //
  // FIXME: There is more per-file stuff we could just drop here?
  if (CI.getFrontendOpts().DisableFree) {
    CI.takeASTConsumer();
    if (!isCurrentFileAST()) {
      CI.takeSema();
      CI.resetAndLeakASTContext();
    }
  } else {
    if (!isCurrentFileAST()) {
      CI.setSema(0);
      CI.setASTContext(0);
    }
    CI.setASTConsumer(0);
  }

  // Inform the preprocessor we are done.
  if (CI.hasPreprocessor())
    CI.getPreprocessor().EndSourceFile();

  if (CI.getFrontendOpts().ShowStats) {
    llvm::errs() << "\nSTATISTICS FOR '" << getCurrentFile() << "':\n";
    CI.getPreprocessor().PrintStats();
    CI.getPreprocessor().getIdentifierTable().PrintStats();
    CI.getPreprocessor().getHeaderSearchInfo().PrintStats();
    CI.getSourceManager().PrintStats();
    llvm::errs() << "\n";
  }

  // Cleanup the output streams, and erase the output files if we encountered
  // an error.
  CI.clearOutputFiles(/*EraseFiles=*/CI.getDiagnostics().hasErrorOccurred());

  if (isCurrentFileAST()) {
    CI.takeSema();
    CI.resetAndLeakASTContext();
    CI.resetAndLeakPreprocessor();
    CI.resetAndLeakSourceManager();
    CI.resetAndLeakFileManager();
  }

  setCompilerInstance(0);
  setCurrentInput(FrontendInputFile());
}

//===----------------------------------------------------------------------===//
// Utility Actions
//===----------------------------------------------------------------------===//

void ASTFrontendAction::ExecuteAction() {
  CompilerInstance &CI = getCompilerInstance();

  // FIXME: Move the truncation aspect of this into Sema, we delayed this till
  // here so the source manager would be initialized.
  if (hasCodeCompletionSupport() &&
      !CI.getFrontendOpts().CodeCompletionAt.FileName.empty())
    CI.createCodeCompletionConsumer();

  // Use a code completion consumer?
  CodeCompleteConsumer *CompletionConsumer = 0;
  if (CI.hasCodeCompletionConsumer())
    CompletionConsumer = &CI.getCodeCompletionConsumer();

  if (!CI.hasSema())
    CI.createSema(getTranslationUnitKind(), CompletionConsumer);

  ParseAST(CI.getSema(), CI.getFrontendOpts().ShowStats,
           CI.getFrontendOpts().SkipFunctionBodies);
}

void PluginASTAction::anchor() { }

ASTConsumer *
PreprocessorFrontendAction::CreateASTConsumer(CompilerInstance &CI,
                                              StringRef InFile) {
  llvm_unreachable("Invalid CreateASTConsumer on preprocessor action!");
}

ASTConsumer *WrapperFrontendAction::CreateASTConsumer(CompilerInstance &CI,
                                                      StringRef InFile) {
  return WrappedAction->CreateASTConsumer(CI, InFile);
}
bool WrapperFrontendAction::BeginInvocation(CompilerInstance &CI) {
  return WrappedAction->BeginInvocation(CI);
}
bool WrapperFrontendAction::BeginSourceFileAction(CompilerInstance &CI,
                                                  StringRef Filename) {
  WrappedAction->setCurrentInput(getCurrentInput());
  WrappedAction->setCompilerInstance(&CI);
  return WrappedAction->BeginSourceFileAction(CI, Filename);
}
void WrapperFrontendAction::ExecuteAction() {
  WrappedAction->ExecuteAction();
}
void WrapperFrontendAction::EndSourceFileAction() {
  WrappedAction->EndSourceFileAction();
}

bool WrapperFrontendAction::usesPreprocessorOnly() const {
  return WrappedAction->usesPreprocessorOnly();
}
TranslationUnitKind WrapperFrontendAction::getTranslationUnitKind() {
  return WrappedAction->getTranslationUnitKind();
}
bool WrapperFrontendAction::hasPCHSupport() const {
  return WrappedAction->hasPCHSupport();
}
bool WrapperFrontendAction::hasASTFileSupport() const {
  return WrappedAction->hasASTFileSupport();
}
bool WrapperFrontendAction::hasIRSupport() const {
  return WrappedAction->hasIRSupport();
}
bool WrapperFrontendAction::hasCodeCompletionSupport() const {
  return WrappedAction->hasCodeCompletionSupport();
}

WrapperFrontendAction::WrapperFrontendAction(FrontendAction *WrappedAction)
  : WrappedAction(WrappedAction) {}

