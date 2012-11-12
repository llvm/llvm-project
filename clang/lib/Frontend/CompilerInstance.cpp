//===--- CompilerInstance.cpp ---------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/CompilerInstance.h"
#include "clang/Sema/Sema.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/Version.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/PTHManager.h"
#include "clang/Frontend/ChainedDiagnosticConsumer.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Frontend/LogDiagnosticPrinter.h"
#include "clang/Frontend/SerializedDiagnosticPrinter.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Frontend/VerifyDiagnosticConsumer.h"
#include "clang/Frontend/Utils.h"
#include "clang/Serialization/ASTReader.h"
#include "clang/Sema/CodeCompleteConsumer.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/LockFileManager.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/system_error.h"
#include "llvm/Support/CrashRecoveryContext.h"
#include "llvm/Config/config.h"

using namespace clang;

CompilerInstance::CompilerInstance()
  : Invocation(new CompilerInvocation()), ModuleManager(0) {
}

CompilerInstance::~CompilerInstance() {
  assert(OutputFiles.empty() && "Still output files in flight?");
}

void CompilerInstance::setInvocation(CompilerInvocation *Value) {
  Invocation = Value;
}

void CompilerInstance::setDiagnostics(DiagnosticsEngine *Value) {
  Diagnostics = Value;
}

void CompilerInstance::setTarget(TargetInfo *Value) {
  Target = Value;
}

void CompilerInstance::setFileManager(FileManager *Value) {
  FileMgr = Value;
}

void CompilerInstance::setSourceManager(SourceManager *Value) {
  SourceMgr = Value;
}

void CompilerInstance::setPreprocessor(Preprocessor *Value) { PP = Value; }

void CompilerInstance::setASTContext(ASTContext *Value) { Context = Value; }

void CompilerInstance::setSema(Sema *S) {
  TheSema.reset(S);
}

void CompilerInstance::setASTConsumer(ASTConsumer *Value) {
  Consumer.reset(Value);
}

void CompilerInstance::setCodeCompletionConsumer(CodeCompleteConsumer *Value) {
  CompletionConsumer.reset(Value);
}

// Diagnostics
static void SetUpBuildDumpLog(DiagnosticOptions *DiagOpts,
                              unsigned argc, const char* const *argv,
                              DiagnosticsEngine &Diags) {
  std::string ErrorInfo;
  OwningPtr<raw_ostream> OS(
    new llvm::raw_fd_ostream(DiagOpts->DumpBuildInformation.c_str(),ErrorInfo));
  if (!ErrorInfo.empty()) {
    Diags.Report(diag::err_fe_unable_to_open_logfile)
                 << DiagOpts->DumpBuildInformation << ErrorInfo;
    return;
  }

  (*OS) << "clang -cc1 command line arguments: ";
  for (unsigned i = 0; i != argc; ++i)
    (*OS) << argv[i] << ' ';
  (*OS) << '\n';

  // Chain in a diagnostic client which will log the diagnostics.
  DiagnosticConsumer *Logger =
    new TextDiagnosticPrinter(*OS.take(), DiagOpts, /*OwnsOutputStream=*/true);
  Diags.setClient(new ChainedDiagnosticConsumer(Diags.takeClient(), Logger));
}

static void SetUpDiagnosticLog(DiagnosticOptions *DiagOpts,
                               const CodeGenOptions *CodeGenOpts,
                               DiagnosticsEngine &Diags) {
  std::string ErrorInfo;
  bool OwnsStream = false;
  raw_ostream *OS = &llvm::errs();
  if (DiagOpts->DiagnosticLogFile != "-") {
    // Create the output stream.
    llvm::raw_fd_ostream *FileOS(
      new llvm::raw_fd_ostream(DiagOpts->DiagnosticLogFile.c_str(),
                               ErrorInfo, llvm::raw_fd_ostream::F_Append));
    if (!ErrorInfo.empty()) {
      Diags.Report(diag::warn_fe_cc_log_diagnostics_failure)
        << DiagOpts->DumpBuildInformation << ErrorInfo;
    } else {
      FileOS->SetUnbuffered();
      FileOS->SetUseAtomicWrites(true);
      OS = FileOS;
      OwnsStream = true;
    }
  }

  // Chain in the diagnostic client which will log the diagnostics.
  LogDiagnosticPrinter *Logger = new LogDiagnosticPrinter(*OS, DiagOpts,
                                                          OwnsStream);
  if (CodeGenOpts)
    Logger->setDwarfDebugFlags(CodeGenOpts->DwarfDebugFlags);
  Diags.setClient(new ChainedDiagnosticConsumer(Diags.takeClient(), Logger));
}

static void SetupSerializedDiagnostics(DiagnosticOptions *DiagOpts,
                                       DiagnosticsEngine &Diags,
                                       StringRef OutputFile) {
  std::string ErrorInfo;
  OwningPtr<llvm::raw_fd_ostream> OS;
  OS.reset(new llvm::raw_fd_ostream(OutputFile.str().c_str(), ErrorInfo,
                                    llvm::raw_fd_ostream::F_Binary));
  
  if (!ErrorInfo.empty()) {
    Diags.Report(diag::warn_fe_serialized_diag_failure)
      << OutputFile << ErrorInfo;
    return;
  }
  
  DiagnosticConsumer *SerializedConsumer =
    clang::serialized_diags::create(OS.take(), DiagOpts);

  
  Diags.setClient(new ChainedDiagnosticConsumer(Diags.takeClient(),
                                                SerializedConsumer));
}

void CompilerInstance::createDiagnostics(int Argc, const char* const *Argv,
                                         DiagnosticConsumer *Client,
                                         bool ShouldOwnClient,
                                         bool ShouldCloneClient) {
  Diagnostics = createDiagnostics(&getDiagnosticOpts(), Argc, Argv, Client,
                                  ShouldOwnClient, ShouldCloneClient,
                                  &getCodeGenOpts());
}

IntrusiveRefCntPtr<DiagnosticsEngine>
CompilerInstance::createDiagnostics(DiagnosticOptions *Opts,
                                    int Argc, const char* const *Argv,
                                    DiagnosticConsumer *Client,
                                    bool ShouldOwnClient,
                                    bool ShouldCloneClient,
                                    const CodeGenOptions *CodeGenOpts) {
  IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());
  IntrusiveRefCntPtr<DiagnosticsEngine>
      Diags(new DiagnosticsEngine(DiagID, Opts));

  // Create the diagnostic client for reporting errors or for
  // implementing -verify.
  if (Client) {
    if (ShouldCloneClient)
      Diags->setClient(Client->clone(*Diags), ShouldOwnClient);
    else
      Diags->setClient(Client, ShouldOwnClient);
  } else
    Diags->setClient(new TextDiagnosticPrinter(llvm::errs(), Opts));

  // Chain in -verify checker, if requested.
  if (Opts->VerifyDiagnostics)
    Diags->setClient(new VerifyDiagnosticConsumer(*Diags));

  // Chain in -diagnostic-log-file dumper, if requested.
  if (!Opts->DiagnosticLogFile.empty())
    SetUpDiagnosticLog(Opts, CodeGenOpts, *Diags);

  if (!Opts->DumpBuildInformation.empty())
    SetUpBuildDumpLog(Opts, Argc, Argv, *Diags);

  if (!Opts->DiagnosticSerializationFile.empty())
    SetupSerializedDiagnostics(Opts, *Diags,
                               Opts->DiagnosticSerializationFile);
  
  // Configure our handling of diagnostics.
  ProcessWarningOptions(*Diags, *Opts);

  return Diags;
}

// File Manager

void CompilerInstance::createFileManager() {
  FileMgr = new FileManager(getFileSystemOpts());
}

// Source Manager

void CompilerInstance::createSourceManager(FileManager &FileMgr) {
  SourceMgr = new SourceManager(getDiagnostics(), FileMgr);
}

// Preprocessor

void CompilerInstance::createPreprocessor() {
  const PreprocessorOptions &PPOpts = getPreprocessorOpts();

  // Create a PTH manager if we are using some form of a token cache.
  PTHManager *PTHMgr = 0;
  if (!PPOpts.TokenCache.empty())
    PTHMgr = PTHManager::Create(PPOpts.TokenCache, getDiagnostics());

  // Create the Preprocessor.
  HeaderSearch *HeaderInfo = new HeaderSearch(&getHeaderSearchOpts(),
                                              getFileManager(),
                                              getDiagnostics(),
                                              getLangOpts(),
                                              &getTarget());
  PP = new Preprocessor(&getPreprocessorOpts(),
                        getDiagnostics(), getLangOpts(), &getTarget(),
                        getSourceManager(), *HeaderInfo, *this, PTHMgr,
                        /*OwnsHeaderSearch=*/true);

  // Note that this is different then passing PTHMgr to Preprocessor's ctor.
  // That argument is used as the IdentifierInfoLookup argument to
  // IdentifierTable's ctor.
  if (PTHMgr) {
    PTHMgr->setPreprocessor(&*PP);
    PP->setPTHManager(PTHMgr);
  }

  if (PPOpts.DetailedRecord)
    PP->createPreprocessingRecord(PPOpts.DetailedRecordConditionalDirectives);

  InitializePreprocessor(*PP, PPOpts, getHeaderSearchOpts(), getFrontendOpts());

  // Set up the module path, including the hash for the
  // module-creation options.
  SmallString<256> SpecificModuleCache(
                           getHeaderSearchOpts().ModuleCachePath);
  if (!getHeaderSearchOpts().DisableModuleHash)
    llvm::sys::path::append(SpecificModuleCache,
                            getInvocation().getModuleHash());
  PP->getHeaderSearchInfo().setModuleCachePath(SpecificModuleCache);

  // Handle generating dependencies, if requested.
  const DependencyOutputOptions &DepOpts = getDependencyOutputOpts();
  if (!DepOpts.OutputFile.empty())
    AttachDependencyFileGen(*PP, DepOpts);
  if (!DepOpts.DOTOutputFile.empty())
    AttachDependencyGraphGen(*PP, DepOpts.DOTOutputFile,
                             getHeaderSearchOpts().Sysroot);

  
  // Handle generating header include information, if requested.
  if (DepOpts.ShowHeaderIncludes)
    AttachHeaderIncludeGen(*PP);
  if (!DepOpts.HeaderIncludeOutputFile.empty()) {
    StringRef OutputPath = DepOpts.HeaderIncludeOutputFile;
    if (OutputPath == "-")
      OutputPath = "";
    AttachHeaderIncludeGen(*PP, /*ShowAllHeaders=*/true, OutputPath,
                           /*ShowDepth=*/false);
  }
}

// ASTContext

void CompilerInstance::createASTContext() {
  Preprocessor &PP = getPreprocessor();
  Context = new ASTContext(getLangOpts(), PP.getSourceManager(),
                           &getTarget(), PP.getIdentifierTable(),
                           PP.getSelectorTable(), PP.getBuiltinInfo(),
                           /*size_reserve=*/ 0);
}

// ExternalASTSource

void CompilerInstance::createPCHExternalASTSource(StringRef Path,
                                                  bool DisablePCHValidation,
                                                bool AllowPCHWithCompilerErrors,
                                                 void *DeserializationListener){
  OwningPtr<ExternalASTSource> Source;
  bool Preamble = getPreprocessorOpts().PrecompiledPreambleBytes.first != 0;
  Source.reset(createPCHExternalASTSource(Path, getHeaderSearchOpts().Sysroot,
                                          DisablePCHValidation,
                                          AllowPCHWithCompilerErrors,
                                          getPreprocessor(), getASTContext(),
                                          DeserializationListener,
                                          Preamble));
  ModuleManager = static_cast<ASTReader*>(Source.get());
  getASTContext().setExternalSource(Source);
}

ExternalASTSource *
CompilerInstance::createPCHExternalASTSource(StringRef Path,
                                             const std::string &Sysroot,
                                             bool DisablePCHValidation,
                                             bool AllowPCHWithCompilerErrors,
                                             Preprocessor &PP,
                                             ASTContext &Context,
                                             void *DeserializationListener,
                                             bool Preamble) {
  OwningPtr<ASTReader> Reader;
  Reader.reset(new ASTReader(PP, Context,
                             Sysroot.empty() ? "" : Sysroot.c_str(),
                             DisablePCHValidation,
                             AllowPCHWithCompilerErrors));

  Reader->setDeserializationListener(
            static_cast<ASTDeserializationListener *>(DeserializationListener));
  switch (Reader->ReadAST(Path,
                          Preamble ? serialization::MK_Preamble
                                   : serialization::MK_PCH,
                          ASTReader::ARR_None)) {
  case ASTReader::Success:
    // Set the predefines buffer as suggested by the PCH reader. Typically, the
    // predefines buffer will be empty.
    PP.setPredefines(Reader->getSuggestedPredefines());
    return Reader.take();

  case ASTReader::Failure:
    // Unrecoverable failure: don't even try to process the input file.
    break;

  case ASTReader::OutOfDate:
  case ASTReader::VersionMismatch:
  case ASTReader::ConfigurationMismatch:
  case ASTReader::HadErrors:
    // No suitable PCH file could be found. Return an error.
    break;
  }

  return 0;
}

// Code Completion

static bool EnableCodeCompletion(Preprocessor &PP,
                                 const std::string &Filename,
                                 unsigned Line,
                                 unsigned Column) {
  // Tell the source manager to chop off the given file at a specific
  // line and column.
  const FileEntry *Entry = PP.getFileManager().getFile(Filename);
  if (!Entry) {
    PP.getDiagnostics().Report(diag::err_fe_invalid_code_complete_file)
      << Filename;
    return true;
  }

  // Truncate the named file at the given line/column.
  PP.SetCodeCompletionPoint(Entry, Line, Column);
  return false;
}

void CompilerInstance::createCodeCompletionConsumer() {
  const ParsedSourceLocation &Loc = getFrontendOpts().CodeCompletionAt;
  if (!CompletionConsumer) {
    setCodeCompletionConsumer(
      createCodeCompletionConsumer(getPreprocessor(),
                                   Loc.FileName, Loc.Line, Loc.Column,
                                   getFrontendOpts().CodeCompleteOpts,
                                   llvm::outs()));
    if (!CompletionConsumer)
      return;
  } else if (EnableCodeCompletion(getPreprocessor(), Loc.FileName,
                                  Loc.Line, Loc.Column)) {
    setCodeCompletionConsumer(0);
    return;
  }

  if (CompletionConsumer->isOutputBinary() &&
      llvm::sys::Program::ChangeStdoutToBinary()) {
    getPreprocessor().getDiagnostics().Report(diag::err_fe_stdout_binary);
    setCodeCompletionConsumer(0);
  }
}

void CompilerInstance::createFrontendTimer() {
  FrontendTimer.reset(new llvm::Timer("Clang front-end timer"));
}

CodeCompleteConsumer *
CompilerInstance::createCodeCompletionConsumer(Preprocessor &PP,
                                               const std::string &Filename,
                                               unsigned Line,
                                               unsigned Column,
                                               const CodeCompleteOptions &Opts,
                                               raw_ostream &OS) {
  if (EnableCodeCompletion(PP, Filename, Line, Column))
    return 0;

  // Set up the creation routine for code-completion.
  return new PrintingCodeCompleteConsumer(Opts, OS);
}

void CompilerInstance::createSema(TranslationUnitKind TUKind,
                                  CodeCompleteConsumer *CompletionConsumer) {
  TheSema.reset(new Sema(getPreprocessor(), getASTContext(), getASTConsumer(),
                         TUKind, CompletionConsumer));
}

// Output Files

void CompilerInstance::addOutputFile(const OutputFile &OutFile) {
  assert(OutFile.OS && "Attempt to add empty stream to output list!");
  OutputFiles.push_back(OutFile);
}

void CompilerInstance::clearOutputFiles(bool EraseFiles) {
  for (std::list<OutputFile>::iterator
         it = OutputFiles.begin(), ie = OutputFiles.end(); it != ie; ++it) {
    delete it->OS;
    if (!it->TempFilename.empty()) {
      if (EraseFiles) {
        bool existed;
        llvm::sys::fs::remove(it->TempFilename, existed);
      } else {
        SmallString<128> NewOutFile(it->Filename);

        // If '-working-directory' was passed, the output filename should be
        // relative to that.
        FileMgr->FixupRelativePath(NewOutFile);
        if (llvm::error_code ec = llvm::sys::fs::rename(it->TempFilename,
                                                        NewOutFile.str())) {
          getDiagnostics().Report(diag::err_unable_to_rename_temp)
            << it->TempFilename << it->Filename << ec.message();

          bool existed;
          llvm::sys::fs::remove(it->TempFilename, existed);
        }
      }
    } else if (!it->Filename.empty() && EraseFiles)
      llvm::sys::Path(it->Filename).eraseFromDisk();

  }
  OutputFiles.clear();
}

llvm::raw_fd_ostream *
CompilerInstance::createDefaultOutputFile(bool Binary,
                                          StringRef InFile,
                                          StringRef Extension) {
  return createOutputFile(getFrontendOpts().OutputFile, Binary,
                          /*RemoveFileOnSignal=*/true, InFile, Extension,
                          /*UseTemporary=*/true);
}

llvm::raw_fd_ostream *
CompilerInstance::createOutputFile(StringRef OutputPath,
                                   bool Binary, bool RemoveFileOnSignal,
                                   StringRef InFile,
                                   StringRef Extension,
                                   bool UseTemporary,
                                   bool CreateMissingDirectories) {
  std::string Error, OutputPathName, TempPathName;
  llvm::raw_fd_ostream *OS = createOutputFile(OutputPath, Error, Binary,
                                              RemoveFileOnSignal,
                                              InFile, Extension,
                                              UseTemporary,
                                              CreateMissingDirectories,
                                              &OutputPathName,
                                              &TempPathName);
  if (!OS) {
    getDiagnostics().Report(diag::err_fe_unable_to_open_output)
      << OutputPath << Error;
    return 0;
  }

  // Add the output file -- but don't try to remove "-", since this means we are
  // using stdin.
  addOutputFile(OutputFile((OutputPathName != "-") ? OutputPathName : "",
                TempPathName, OS));

  return OS;
}

llvm::raw_fd_ostream *
CompilerInstance::createOutputFile(StringRef OutputPath,
                                   std::string &Error,
                                   bool Binary,
                                   bool RemoveFileOnSignal,
                                   StringRef InFile,
                                   StringRef Extension,
                                   bool UseTemporary,
                                   bool CreateMissingDirectories,
                                   std::string *ResultPathName,
                                   std::string *TempPathName) {
  assert((!CreateMissingDirectories || UseTemporary) &&
         "CreateMissingDirectories is only allowed when using temporary files");

  std::string OutFile, TempFile;
  if (!OutputPath.empty()) {
    OutFile = OutputPath;
  } else if (InFile == "-") {
    OutFile = "-";
  } else if (!Extension.empty()) {
    llvm::sys::Path Path(InFile);
    Path.eraseSuffix();
    Path.appendSuffix(Extension);
    OutFile = Path.str();
  } else {
    OutFile = "-";
  }

  OwningPtr<llvm::raw_fd_ostream> OS;
  std::string OSFile;

  if (UseTemporary && OutFile != "-") {
    // Only create the temporary if the parent directory exists (or create
    // missing directories is true) and we can actually write to OutPath,
    // otherwise we want to fail early.
    SmallString<256> AbsPath(OutputPath);
    llvm::sys::fs::make_absolute(AbsPath);
    llvm::sys::Path OutPath(AbsPath);
    bool ParentExists = false;
    if (llvm::sys::fs::exists(llvm::sys::path::parent_path(AbsPath.str()),
                              ParentExists))
      ParentExists = false;
    bool Exists;
    if ((CreateMissingDirectories || ParentExists) &&
        ((llvm::sys::fs::exists(AbsPath.str(), Exists) || !Exists) ||
         (OutPath.isRegularFile() && OutPath.canWrite()))) {
      // Create a temporary file.
      SmallString<128> TempPath;
      TempPath = OutFile;
      TempPath += "-%%%%%%%%";
      int fd;
      if (llvm::sys::fs::unique_file(TempPath.str(), fd, TempPath,
                                     /*makeAbsolute=*/false, 0664)
          == llvm::errc::success) {
        OS.reset(new llvm::raw_fd_ostream(fd, /*shouldClose=*/true));
        OSFile = TempFile = TempPath.str();
      }
    }
  }

  if (!OS) {
    OSFile = OutFile;
    OS.reset(
      new llvm::raw_fd_ostream(OSFile.c_str(), Error,
                               (Binary ? llvm::raw_fd_ostream::F_Binary : 0)));
    if (!Error.empty())
      return 0;
  }

  // Make sure the out stream file gets removed if we crash.
  if (RemoveFileOnSignal)
    llvm::sys::RemoveFileOnSignal(llvm::sys::Path(OSFile));

  if (ResultPathName)
    *ResultPathName = OutFile;
  if (TempPathName)
    *TempPathName = TempFile;

  return OS.take();
}

// Initialization Utilities

bool CompilerInstance::InitializeSourceManager(const FrontendInputFile &Input){
  return InitializeSourceManager(Input, getDiagnostics(),
                                 getFileManager(), getSourceManager(), 
                                 getFrontendOpts());
}

bool CompilerInstance::InitializeSourceManager(const FrontendInputFile &Input,
                                               DiagnosticsEngine &Diags,
                                               FileManager &FileMgr,
                                               SourceManager &SourceMgr,
                                               const FrontendOptions &Opts) {
  SrcMgr::CharacteristicKind
    Kind = Input.isSystem() ? SrcMgr::C_System : SrcMgr::C_User;

  if (Input.isBuffer()) {
    SourceMgr.createMainFileIDForMemBuffer(Input.getBuffer(), Kind);
    assert(!SourceMgr.getMainFileID().isInvalid() &&
           "Couldn't establish MainFileID!");
    return true;
  }

  StringRef InputFile = Input.getFile();

  // Figure out where to get and map in the main file.
  if (InputFile != "-") {
    const FileEntry *File = FileMgr.getFile(InputFile);
    if (!File) {
      Diags.Report(diag::err_fe_error_reading) << InputFile;
      return false;
    }
    SourceMgr.createMainFileID(File, Kind);

    // The natural SourceManager infrastructure can't currently handle named
    // pipes, but we would at least like to accept them for the main
    // file. Detect them here, read them with the more generic MemoryBuffer
    // function, and simply override their contents as we do for STDIN.
    if (File->isNamedPipe()) {
      OwningPtr<llvm::MemoryBuffer> MB;
      if (llvm::error_code ec = llvm::MemoryBuffer::getFile(InputFile, MB)) {
        Diags.Report(diag::err_cannot_open_file) << InputFile << ec.message();
        return false;
      }
      SourceMgr.overrideFileContents(File, MB.take());
    }
  } else {
    OwningPtr<llvm::MemoryBuffer> SB;
    if (llvm::MemoryBuffer::getSTDIN(SB)) {
      // FIXME: Give ec.message() in this diag.
      Diags.Report(diag::err_fe_error_reading_stdin);
      return false;
    }
    const FileEntry *File = FileMgr.getVirtualFile(SB->getBufferIdentifier(),
                                                   SB->getBufferSize(), 0);
    SourceMgr.createMainFileID(File, Kind);
    SourceMgr.overrideFileContents(File, SB.take());
  }

  assert(!SourceMgr.getMainFileID().isInvalid() &&
         "Couldn't establish MainFileID!");
  return true;
}

// High-Level Operations

bool CompilerInstance::ExecuteAction(FrontendAction &Act) {
  assert(hasDiagnostics() && "Diagnostics engine is not initialized!");
  assert(!getFrontendOpts().ShowHelp && "Client must handle '-help'!");
  assert(!getFrontendOpts().ShowVersion && "Client must handle '-version'!");

  // FIXME: Take this as an argument, once all the APIs we used have moved to
  // taking it as an input instead of hard-coding llvm::errs.
  raw_ostream &OS = llvm::errs();

  // Create the target instance.
  setTarget(TargetInfo::CreateTargetInfo(getDiagnostics(), getTargetOpts()));
  if (!hasTarget())
    return false;

  // Inform the target of the language options.
  //
  // FIXME: We shouldn't need to do this, the target should be immutable once
  // created. This complexity should be lifted elsewhere.
  getTarget().setForcedLangOptions(getLangOpts());

  // rewriter project will change target built-in bool type from its default. 
  if (getFrontendOpts().ProgramAction == frontend::RewriteObjC)
    getTarget().noSignedCharForObjCBool();

  // Validate/process some options.
  if (getHeaderSearchOpts().Verbose)
    OS << "clang -cc1 version " CLANG_VERSION_STRING
       << " based upon " << PACKAGE_STRING
       << " default target " << llvm::sys::getDefaultTargetTriple() << "\n";

  if (getFrontendOpts().ShowTimers)
    createFrontendTimer();

  if (getFrontendOpts().ShowStats)
    llvm::EnableStatistics();

  for (unsigned i = 0, e = getFrontendOpts().Inputs.size(); i != e; ++i) {
    // Reset the ID tables if we are reusing the SourceManager.
    if (hasSourceManager())
      getSourceManager().clearIDTables();

    if (Act.BeginSourceFile(*this, getFrontendOpts().Inputs[i])) {
      Act.Execute();
      Act.EndSourceFile();
    }
  }

  // Notify the diagnostic client that all files were processed.
  getDiagnostics().getClient()->finish();

  if (getDiagnosticOpts().ShowCarets) {
    // We can have multiple diagnostics sharing one diagnostic client.
    // Get the total number of warnings/errors from the client.
    unsigned NumWarnings = getDiagnostics().getClient()->getNumWarnings();
    unsigned NumErrors = getDiagnostics().getClient()->getNumErrors();

    if (NumWarnings)
      OS << NumWarnings << " warning" << (NumWarnings == 1 ? "" : "s");
    if (NumWarnings && NumErrors)
      OS << " and ";
    if (NumErrors)
      OS << NumErrors << " error" << (NumErrors == 1 ? "" : "s");
    if (NumWarnings || NumErrors)
      OS << " generated.\n";
  }

  if (getFrontendOpts().ShowStats && hasFileManager()) {
    getFileManager().PrintStats();
    OS << "\n";
  }

  return !getDiagnostics().getClient()->getNumErrors();
}

/// \brief Determine the appropriate source input kind based on language
/// options.
static InputKind getSourceInputKindFromOptions(const LangOptions &LangOpts) {
  if (LangOpts.OpenCL)
    return IK_OpenCL;
  if (LangOpts.CUDA)
    return IK_CUDA;
  if (LangOpts.ObjC1)
    return LangOpts.CPlusPlus? IK_ObjCXX : IK_ObjC;
  return LangOpts.CPlusPlus? IK_CXX : IK_C;
}

namespace {
  struct CompileModuleMapData {
    CompilerInstance &Instance;
    GenerateModuleAction &CreateModuleAction;
  };
}

/// \brief Helper function that executes the module-generating action under
/// a crash recovery context.
static void doCompileMapModule(void *UserData) {
  CompileModuleMapData &Data
    = *reinterpret_cast<CompileModuleMapData *>(UserData);
  Data.Instance.ExecuteAction(Data.CreateModuleAction);
}

/// \brief Compile a module file for the given module, using the options 
/// provided by the importing compiler instance.
static void compileModule(CompilerInstance &ImportingInstance,
                          Module *Module,
                          StringRef ModuleFileName) {
  llvm::LockFileManager Locked(ModuleFileName);
  switch (Locked) {
  case llvm::LockFileManager::LFS_Error:
    return;

  case llvm::LockFileManager::LFS_Owned:
    // We're responsible for building the module ourselves. Do so below.
    break;

  case llvm::LockFileManager::LFS_Shared:
    // Someone else is responsible for building the module. Wait for them to
    // finish.
    Locked.waitForUnlock();
    return;
  }

  ModuleMap &ModMap 
    = ImportingInstance.getPreprocessor().getHeaderSearchInfo().getModuleMap();
    
  // Construct a compiler invocation for creating this module.
  IntrusiveRefCntPtr<CompilerInvocation> Invocation
    (new CompilerInvocation(ImportingInstance.getInvocation()));

  PreprocessorOptions &PPOpts = Invocation->getPreprocessorOpts();
  
  // For any options that aren't intended to affect how a module is built,
  // reset them to their default values.
  Invocation->getLangOpts()->resetNonModularOptions();
  PPOpts.resetNonModularOptions();

  // Note the name of the module we're building.
  Invocation->getLangOpts()->CurrentModule = Module->getTopLevelModuleName();

  // Note that this module is part of the module build path, so that we
  // can detect cycles in the module graph.
  PPOpts.ModuleBuildPath.push_back(Module->getTopLevelModuleName());

  // If there is a module map file, build the module using the module map.
  // Set up the inputs/outputs so that we build the module from its umbrella
  // header.
  FrontendOptions &FrontendOpts = Invocation->getFrontendOpts();
  FrontendOpts.OutputFile = ModuleFileName.str();
  FrontendOpts.DisableFree = false;
  FrontendOpts.Inputs.clear();
  InputKind IK = getSourceInputKindFromOptions(*Invocation->getLangOpts());

  // Get or create the module map that we'll use to build this module.
  SmallString<128> TempModuleMapFileName;
  if (const FileEntry *ModuleMapFile
                                  = ModMap.getContainingModuleMapFile(Module)) {
    // Use the module map where this module resides.
    FrontendOpts.Inputs.push_back(FrontendInputFile(ModuleMapFile->getName(), 
                                                    IK));
  } else {
    // Create a temporary module map file.
    TempModuleMapFileName = Module->Name;
    TempModuleMapFileName += "-%%%%%%%%.map";
    int FD;
    if (llvm::sys::fs::unique_file(TempModuleMapFileName.str(), FD, 
                                   TempModuleMapFileName,
                                   /*makeAbsolute=*/true)
          != llvm::errc::success) {
      ImportingInstance.getDiagnostics().Report(diag::err_module_map_temp_file)
        << TempModuleMapFileName;
      return;
    }
    // Print the module map to this file.
    llvm::raw_fd_ostream OS(FD, /*shouldClose=*/true);
    Module->print(OS);
    FrontendOpts.Inputs.push_back(
      FrontendInputFile(TempModuleMapFileName.str().str(), IK));
  }

  // Don't free the remapped file buffers; they are owned by our caller.
  PPOpts.RetainRemappedFileBuffers = true;
    
  Invocation->getDiagnosticOpts().VerifyDiagnostics = 0;
  assert(ImportingInstance.getInvocation().getModuleHash() ==
         Invocation->getModuleHash() && "Module hash mismatch!");
  
  // Construct a compiler instance that will be used to actually create the
  // module.
  CompilerInstance Instance;
  Instance.setInvocation(&*Invocation);
  Instance.createDiagnostics(/*argc=*/0, /*argv=*/0,
                             &ImportingInstance.getDiagnosticClient(),
                             /*ShouldOwnClient=*/true,
                             /*ShouldCloneClient=*/true);
  
  // Construct a module-generating action.
  GenerateModuleAction CreateModuleAction;
  
  // Execute the action to actually build the module in-place. Use a separate
  // thread so that we get a stack large enough.
  const unsigned ThreadStackSize = 8 << 20;
  llvm::CrashRecoveryContext CRC;
  CompileModuleMapData Data = { Instance, CreateModuleAction };
  CRC.RunSafelyOnThread(&doCompileMapModule, &Data, ThreadStackSize);
  
  // Delete the temporary module map file.
  // FIXME: Even though we're executing under crash protection, it would still
  // be nice to do this with RemoveFileOnSignal when we can. However, that
  // doesn't make sense for all clients, so clean this up manually.
  Instance.clearOutputFiles(/*EraseFiles=*/true);
  if (!TempModuleMapFileName.empty())
    llvm::sys::Path(TempModuleMapFileName).eraseFromDisk();
}

Module *CompilerInstance::loadModule(SourceLocation ImportLoc, 
                                     ModuleIdPath Path,
                                     Module::NameVisibilityKind Visibility,
                                     bool IsInclusionDirective) {
  // If we've already handled this import, just return the cached result.
  // This one-element cache is important to eliminate redundant diagnostics
  // when both the preprocessor and parser see the same import declaration.
  if (!ImportLoc.isInvalid() && LastModuleImportLoc == ImportLoc) {
    // Make the named module visible.
    if (LastModuleImportResult)
      ModuleManager->makeModuleVisible(LastModuleImportResult, Visibility);
    return LastModuleImportResult;
  }
  
  // Determine what file we're searching from.
  StringRef ModuleName = Path[0].first->getName();
  SourceLocation ModuleNameLoc = Path[0].second;

  clang::Module *Module = 0;
  
  // If we don't already have information on this module, load the module now.
  llvm::DenseMap<const IdentifierInfo *, clang::Module *>::iterator Known
    = KnownModules.find(Path[0].first);
  if (Known != KnownModules.end()) {
    // Retrieve the cached top-level module.
    Module = Known->second;    
  } else if (ModuleName == getLangOpts().CurrentModule) {
    // This is the module we're building. 
    Module = PP->getHeaderSearchInfo().getModuleMap().findModule(ModuleName);
    Known = KnownModules.insert(std::make_pair(Path[0].first, Module)).first;
  } else {
    // Search for a module with the given name.
    Module = PP->getHeaderSearchInfo().lookupModule(ModuleName);
    std::string ModuleFileName;
    if (Module)
      ModuleFileName = PP->getHeaderSearchInfo().getModuleFileName(Module);
    else
      ModuleFileName = PP->getHeaderSearchInfo().getModuleFileName(ModuleName);
    
    if (ModuleFileName.empty()) {
      getDiagnostics().Report(ModuleNameLoc, diag::err_module_not_found)
        << ModuleName
        << SourceRange(ImportLoc, ModuleNameLoc);
      LastModuleImportLoc = ImportLoc;
      LastModuleImportResult = 0;
      return 0;
    }
    
    const FileEntry *ModuleFile
      = getFileManager().getFile(ModuleFileName, /*OpenFile=*/false,
                                 /*CacheFailure=*/false);
    bool BuildingModule = false;
    if (!ModuleFile && Module) {
      // The module is not cached, but we have a module map from which we can
      // build the module.

      // Check whether there is a cycle in the module graph.
      SmallVectorImpl<std::string> &ModuleBuildPath
        = getPreprocessorOpts().ModuleBuildPath;
      SmallVectorImpl<std::string>::iterator Pos
        = std::find(ModuleBuildPath.begin(), ModuleBuildPath.end(), ModuleName);
      if (Pos != ModuleBuildPath.end()) {
        SmallString<256> CyclePath;
        for (; Pos != ModuleBuildPath.end(); ++Pos) {
          CyclePath += *Pos;
          CyclePath += " -> ";
        }
        CyclePath += ModuleName;

        getDiagnostics().Report(ModuleNameLoc, diag::err_module_cycle)
          << ModuleName << CyclePath;
        return 0;
      }

      getDiagnostics().Report(ModuleNameLoc, diag::warn_module_build)
        << ModuleName;
      BuildingModule = true;
      compileModule(*this, Module, ModuleFileName);
      ModuleFile = FileMgr->getFile(ModuleFileName);
    }

    if (!ModuleFile) {
      getDiagnostics().Report(ModuleNameLoc,
                              BuildingModule? diag::err_module_not_built
                                            : diag::err_module_not_found)
        << ModuleName
        << SourceRange(ImportLoc, ModuleNameLoc);
      return 0;
    }

    // If we don't already have an ASTReader, create one now.
    if (!ModuleManager) {
      if (!hasASTContext())
        createASTContext();

      std::string Sysroot = getHeaderSearchOpts().Sysroot;
      const PreprocessorOptions &PPOpts = getPreprocessorOpts();
      ModuleManager = new ASTReader(getPreprocessor(), *Context,
                                    Sysroot.empty() ? "" : Sysroot.c_str(),
                                    PPOpts.DisablePCHValidation);
      if (hasASTConsumer()) {
        ModuleManager->setDeserializationListener(
          getASTConsumer().GetASTDeserializationListener());
        getASTContext().setASTMutationListener(
          getASTConsumer().GetASTMutationListener());
        getPreprocessor().setPPMutationListener(
          getASTConsumer().GetPPMutationListener());
      }
      OwningPtr<ExternalASTSource> Source;
      Source.reset(ModuleManager);
      getASTContext().setExternalSource(Source);
      if (hasSema())
        ModuleManager->InitializeSema(getSema());
      if (hasASTConsumer())
        ModuleManager->StartTranslationUnit(&getASTConsumer());
    }

    // Try to load the module we found.
    unsigned ARRFlags = ASTReader::ARR_None;
    if (Module)
      ARRFlags |= ASTReader::ARR_OutOfDate;
    switch (ModuleManager->ReadAST(ModuleFile->getName(),
                                   serialization::MK_Module,
                                   ARRFlags)) {
    case ASTReader::Success:
      break;

    case ASTReader::OutOfDate: {
      // The module file is out-of-date. Rebuild it.
      getFileManager().invalidateCache(ModuleFile);
      bool Existed;
      llvm::sys::fs::remove(ModuleFileName, Existed);
      compileModule(*this, Module, ModuleFileName);

      // Try loading the module again.
      ModuleFile = FileMgr->getFile(ModuleFileName);
      if (!ModuleFile ||
          ModuleManager->ReadAST(ModuleFileName,
                                 serialization::MK_Module,
                                 ASTReader::ARR_None) != ASTReader::Success) {
        KnownModules[Path[0].first] = 0;
        return 0;
      }

      // Okay, we've rebuilt and now loaded the module.
      break;
    }

    case ASTReader::VersionMismatch:
    case ASTReader::ConfigurationMismatch:
    case ASTReader::HadErrors:
      // FIXME: The ASTReader will already have complained, but can we showhorn
      // that diagnostic information into a more useful form?
      KnownModules[Path[0].first] = 0;
      return 0;

    case ASTReader::Failure:
      // Already complained, but note now that we failed.
      KnownModules[Path[0].first] = 0;
      return 0;
    }
    
    if (!Module) {
      // If we loaded the module directly, without finding a module map first,
      // we'll have loaded the module's information from the module itself.
      Module = PP->getHeaderSearchInfo().getModuleMap()
                 .findModule((Path[0].first->getName()));
    }

    if (Module)
      Module->setASTFile(ModuleFile);
    
    // Cache the result of this top-level module lookup for later.
    Known = KnownModules.insert(std::make_pair(Path[0].first, Module)).first;
  }
  
  // If we never found the module, fail.
  if (!Module)
    return 0;
  
  // Verify that the rest of the module path actually corresponds to
  // a submodule.
  if (Path.size() > 1) {
    for (unsigned I = 1, N = Path.size(); I != N; ++I) {
      StringRef Name = Path[I].first->getName();
      clang::Module *Sub = Module->findSubmodule(Name);
      
      if (!Sub) {
        // Attempt to perform typo correction to find a module name that works.
        llvm::SmallVector<StringRef, 2> Best;
        unsigned BestEditDistance = (std::numeric_limits<unsigned>::max)();
        
        for (clang::Module::submodule_iterator J = Module->submodule_begin(), 
                                            JEnd = Module->submodule_end();
             J != JEnd; ++J) {
          unsigned ED = Name.edit_distance((*J)->Name,
                                           /*AllowReplacements=*/true,
                                           BestEditDistance);
          if (ED <= BestEditDistance) {
            if (ED < BestEditDistance) {
              Best.clear();
              BestEditDistance = ED;
            }
            
            Best.push_back((*J)->Name);
          }
        }
        
        // If there was a clear winner, user it.
        if (Best.size() == 1) {
          getDiagnostics().Report(Path[I].second, 
                                  diag::err_no_submodule_suggest)
            << Path[I].first << Module->getFullModuleName() << Best[0]
            << SourceRange(Path[0].second, Path[I-1].second)
            << FixItHint::CreateReplacement(SourceRange(Path[I].second),
                                            Best[0]);
          
          Sub = Module->findSubmodule(Best[0]);
        }
      }
      
      if (!Sub) {
        // No submodule by this name. Complain, and don't look for further
        // submodules.
        getDiagnostics().Report(Path[I].second, diag::err_no_submodule)
          << Path[I].first << Module->getFullModuleName()
          << SourceRange(Path[0].second, Path[I-1].second);
        break;
      }
      
      Module = Sub;
    }
  }
  
  // Make the named module visible, if it's not already part of the module
  // we are parsing.
  if (ModuleName != getLangOpts().CurrentModule) {
    if (!Module->IsFromModuleFile) {
      // We have an umbrella header or directory that doesn't actually include
      // all of the headers within the directory it covers. Complain about
      // this missing submodule and recover by forgetting that we ever saw
      // this submodule.
      // FIXME: Should we detect this at module load time? It seems fairly
      // expensive (and rare).
      getDiagnostics().Report(ImportLoc, diag::warn_missing_submodule)
        << Module->getFullModuleName()
        << SourceRange(Path.front().second, Path.back().second);
      
      return 0;
    }

    // Check whether this module is available.
    StringRef Feature;
    if (!Module->isAvailable(getLangOpts(), getTarget(), Feature)) {
      getDiagnostics().Report(ImportLoc, diag::err_module_unavailable)
        << Module->getFullModuleName()
        << Feature
        << SourceRange(Path.front().second, Path.back().second);
      LastModuleImportLoc = ImportLoc;
      LastModuleImportResult = 0;
      return 0;
    }

    ModuleManager->makeModuleVisible(Module, Visibility);
  }
  
  // If this module import was due to an inclusion directive, create an 
  // implicit import declaration to capture it in the AST.
  if (IsInclusionDirective && hasASTContext()) {
    TranslationUnitDecl *TU = getASTContext().getTranslationUnitDecl();
    ImportDecl *ImportD = ImportDecl::CreateImplicit(getASTContext(), TU,
                                                     ImportLoc, Module,
                                                     Path.back().second);
    TU->addDecl(ImportD);
    if (Consumer)
      Consumer->HandleImplicitImportDecl(ImportD);
  }
  
  LastModuleImportLoc = ImportLoc;
  LastModuleImportResult = Module;
  return Module;
}
