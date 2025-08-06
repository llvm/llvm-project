//===--- CompilerInstance.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/CompilerInstance.h"
#include "clang/APINotes/APINotesReader.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/Basic/CharInfo.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticCAS.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/LangStandard.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/Stack.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/Version.h"
#include "clang/CAS/IncludeTree.h"
#include "clang/Config/config.h"
#include "clang/Frontend/CASDependencyCollector.h"
#include "clang/Frontend/ChainedDiagnosticConsumer.h"
#include "clang/Frontend/CompileJobCacheKey.h"
#include "clang/Frontend/CompileJobCacheResult.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/Frontend/LogDiagnosticPrinter.h"
#include "clang/Frontend/SARIFDiagnosticPrinter.h"
#include "clang/Frontend/SerializedDiagnosticPrinter.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Frontend/Utils.h"
#include "clang/Frontend/VerifyDiagnosticConsumer.h"
#include "clang/Index/IndexingAction.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "clang/Sema/CodeCompleteConsumer.h"
#include "clang/Sema/ParsedAttr.h"
#include "clang/Sema/Sema.h"
#include "clang/Serialization/ASTReader.h"
#include "clang/Serialization/GlobalModuleIndex.h"
#include "clang/Serialization/InMemoryModuleCache.h"
#include "clang/Serialization/ModuleCache.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CAS/ActionCache.h"
#include "llvm/CAS/ObjectStore.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/AdvisoryLock.h"
#include "llvm/Support/BuryPointer.h"
#include "llvm/Support/CrashRecoveryContext.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/TimeProfiler.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Support/VirtualOutputBackends.h"
#include "llvm/Support/VirtualOutputError.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"
#include <optional>
#include <time.h>
#include <utility>

using namespace clang;

CompilerInstance::CompilerInstance(
    std::shared_ptr<CompilerInvocation> Invocation,
    std::shared_ptr<PCHContainerOperations> PCHContainerOps,
    ModuleCache *ModCache)
    : ModuleLoader(/*BuildingModule=*/ModCache),
      Invocation(std::move(Invocation)),
      ModCache(ModCache ? ModCache : createCrossProcessModuleCache()),
      ThePCHContainerOperations(std::move(PCHContainerOps)) {
  assert(this->Invocation && "Invocation must not be null");
}

CompilerInstance::~CompilerInstance() {
  assert(OutputFiles.empty() && "Still output files in flight?");
}

bool CompilerInstance::shouldBuildGlobalModuleIndex() const {
  return (BuildGlobalModuleIndex ||
          (TheASTReader && TheASTReader->isGlobalIndexUnavailable() &&
           getFrontendOpts().GenerateGlobalModuleIndex)) &&
         !DisableGeneratingGlobalModuleIndex;
}

void CompilerInstance::setDiagnostics(
    llvm::IntrusiveRefCntPtr<DiagnosticsEngine> Value) {
  Diagnostics = std::move(Value);
}

bool CompilerInstance::isSourceNonReproducible() const {
  assert(PP && "Need to have preprocessor");
  return PP->isSourceNonReproducible();
}

void CompilerInstance::setVerboseOutputStream(raw_ostream &Value) {
  OwnedVerboseOutputStream.reset();
  VerboseOutputStream = &Value;
}

void CompilerInstance::setVerboseOutputStream(std::unique_ptr<raw_ostream> Value) {
  OwnedVerboseOutputStream.swap(Value);
  VerboseOutputStream = OwnedVerboseOutputStream.get();
}

void CompilerInstance::setTarget(TargetInfo *Value) { Target = Value; }
void CompilerInstance::setAuxTarget(TargetInfo *Value) { AuxTarget = Value; }

bool CompilerInstance::createTarget() {
  // Create the target instance.
  setTarget(TargetInfo::CreateTargetInfo(getDiagnostics(),
                                         getInvocation().getTargetOpts()));
  if (!hasTarget())
    return false;

  // Check whether AuxTarget exists, if not, then create TargetInfo for the
  // other side of CUDA/OpenMP/SYCL compilation.
  if (!getAuxTarget() &&
      (getLangOpts().CUDA || getLangOpts().isTargetDevice()) &&
      !getFrontendOpts().AuxTriple.empty()) {
    auto &TO = AuxTargetOpts = std::make_unique<TargetOptions>();
    TO->Triple = llvm::Triple::normalize(getFrontendOpts().AuxTriple);
    if (getFrontendOpts().AuxTargetCPU)
      TO->CPU = *getFrontendOpts().AuxTargetCPU;
    if (getFrontendOpts().AuxTargetFeatures)
      TO->FeaturesAsWritten = *getFrontendOpts().AuxTargetFeatures;
    TO->HostTriple = getTarget().getTriple().str();
    setAuxTarget(TargetInfo::CreateTargetInfo(getDiagnostics(), *TO));
  }

  if (!getTarget().hasStrictFP() && !getLangOpts().ExpStrictFP) {
    if (getLangOpts().RoundingMath) {
      getDiagnostics().Report(diag::warn_fe_backend_unsupported_fp_rounding);
      getLangOpts().RoundingMath = false;
    }
    auto FPExc = getLangOpts().getFPExceptionMode();
    if (FPExc != LangOptions::FPE_Default && FPExc != LangOptions::FPE_Ignore) {
      getDiagnostics().Report(diag::warn_fe_backend_unsupported_fp_exceptions);
      getLangOpts().setFPExceptionMode(LangOptions::FPE_Ignore);
    }
    // FIXME: can we disable FEnvAccess?
  }

  // We should do it here because target knows nothing about
  // language options when it's being created.
  if (getLangOpts().OpenCL &&
      !getTarget().validateOpenCLTarget(getLangOpts(), getDiagnostics()))
    return false;

  // Inform the target of the language options.
  // FIXME: We shouldn't need to do this, the target should be immutable once
  // created. This complexity should be lifted elsewhere.
  getTarget().adjust(getDiagnostics(), getLangOpts(), getAuxTarget());

  if (auto *Aux = getAuxTarget())
    getTarget().setAuxTarget(Aux);

  return true;
}

llvm::vfs::FileSystem &CompilerInstance::getVirtualFileSystem() const {
  return getFileManager().getVirtualFileSystem();
}

llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem>
CompilerInstance::getVirtualFileSystemPtr() const {
  return getFileManager().getVirtualFileSystemPtr();
}

void CompilerInstance::setFileManager(
    llvm::IntrusiveRefCntPtr<FileManager> Value) {
  FileMgr = std::move(Value);
}

void CompilerInstance::setSourceManager(
    llvm::IntrusiveRefCntPtr<SourceManager> Value) {
  SourceMgr = std::move(Value);
}

void CompilerInstance::setPreprocessor(std::shared_ptr<Preprocessor> Value) {
  PP = std::move(Value);
}

void CompilerInstance::setASTContext(
    llvm::IntrusiveRefCntPtr<ASTContext> Value) {
  Context = std::move(Value);

  if (Context && Consumer)
    getASTConsumer().Initialize(getASTContext());
}

void CompilerInstance::setSema(Sema *S) {
  TheSema.reset(S);
}

void CompilerInstance::setASTConsumer(std::unique_ptr<ASTConsumer> Value) {
  Consumer = std::move(Value);

  if (Context && Consumer)
    getASTConsumer().Initialize(getASTContext());
}

void CompilerInstance::setCodeCompletionConsumer(CodeCompleteConsumer *Value) {
  CompletionConsumer.reset(Value);
}

std::unique_ptr<Sema> CompilerInstance::takeSema() {
  return std::move(TheSema);
}

IntrusiveRefCntPtr<ASTReader> CompilerInstance::getASTReader() const {
  return TheASTReader;
}
void CompilerInstance::setASTReader(IntrusiveRefCntPtr<ASTReader> Reader) {
  assert(ModCache.get() == &Reader->getModuleManager().getModuleCache() &&
         "Expected ASTReader to use the same PCM cache");
  TheASTReader = std::move(Reader);
}

std::shared_ptr<ModuleDependencyCollector>
CompilerInstance::getModuleDepCollector() const {
  return ModuleDepCollector;
}

void CompilerInstance::setModuleDepCollector(
    std::shared_ptr<ModuleDependencyCollector> Collector) {
  ModuleDepCollector = std::move(Collector);
}

static void collectHeaderMaps(const HeaderSearch &HS,
                              std::shared_ptr<ModuleDependencyCollector> MDC) {
  SmallVector<std::string, 4> HeaderMapFileNames;
  HS.getHeaderMapFileNames(HeaderMapFileNames);
  for (auto &Name : HeaderMapFileNames)
    MDC->addFile(Name);
}

static void collectIncludePCH(CompilerInstance &CI,
                              std::shared_ptr<ModuleDependencyCollector> MDC) {
  const PreprocessorOptions &PPOpts = CI.getPreprocessorOpts();
  if (PPOpts.ImplicitPCHInclude.empty())
    return;

  StringRef PCHInclude = PPOpts.ImplicitPCHInclude;
  FileManager &FileMgr = CI.getFileManager();
  auto PCHDir = FileMgr.getOptionalDirectoryRef(PCHInclude);
  if (!PCHDir) {
    MDC->addFile(PCHInclude);
    return;
  }

  std::error_code EC;
  SmallString<128> DirNative;
  llvm::sys::path::native(PCHDir->getName(), DirNative);
  llvm::vfs::FileSystem &FS = FileMgr.getVirtualFileSystem();
  SimpleASTReaderListener Validator(CI.getPreprocessor());
  for (llvm::vfs::directory_iterator Dir = FS.dir_begin(DirNative, EC), DirEnd;
       Dir != DirEnd && !EC; Dir.increment(EC)) {
    // Check whether this is an AST file. ASTReader::isAcceptableASTFile is not
    // used here since we're not interested in validating the PCH at this time,
    // but only to check whether this is a file containing an AST.
    if (!ASTReader::readASTFileControlBlock(
            Dir->path(), FileMgr, CI.getModuleCache(),
            CI.getPCHContainerReader(),
            /*FindModuleFileExtensions=*/false, Validator,
            /*ValidateDiagnosticOptions=*/false))
      MDC->addFile(Dir->path());
  }
}

static void collectVFSEntries(CompilerInstance &CI,
                              std::shared_ptr<ModuleDependencyCollector> MDC) {
  if (CI.getHeaderSearchOpts().VFSOverlayFiles.empty())
    return;

  // Collect all VFS found.
  SmallVector<llvm::vfs::YAMLVFSEntry, 16> VFSEntries;
  for (const std::string &VFSFile : CI.getHeaderSearchOpts().VFSOverlayFiles) {
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> Buffer =
        llvm::MemoryBuffer::getFile(VFSFile);
    if (!Buffer)
      return;
    llvm::vfs::collectVFSFromYAML(std::move(Buffer.get()),
                                  /*DiagHandler*/ nullptr, VFSFile, VFSEntries);
  }

  for (auto &E : VFSEntries)
    MDC->addFile(E.VPath, E.RPath);
}

// Diagnostics
static void SetUpDiagnosticLog(DiagnosticOptions &DiagOpts,
                               const CodeGenOptions *CodeGenOpts,
                               DiagnosticsEngine &Diags) {
  std::error_code EC;
  std::unique_ptr<raw_ostream> StreamOwner;
  raw_ostream *OS = &llvm::errs();
  if (DiagOpts.DiagnosticLogFile != "-") {
    // Create the output stream.
    auto FileOS = std::make_unique<llvm::raw_fd_ostream>(
        DiagOpts.DiagnosticLogFile, EC,
        llvm::sys::fs::OF_Append | llvm::sys::fs::OF_TextWithCRLF);
    if (EC) {
      Diags.Report(diag::warn_fe_cc_log_diagnostics_failure)
          << DiagOpts.DiagnosticLogFile << EC.message();
    } else {
      FileOS->SetUnbuffered();
      OS = FileOS.get();
      StreamOwner = std::move(FileOS);
    }
  }

  // Chain in the diagnostic client which will log the diagnostics.
  auto Logger = std::make_unique<LogDiagnosticPrinter>(*OS, DiagOpts,
                                                        std::move(StreamOwner));
  if (CodeGenOpts)
    Logger->setDwarfDebugFlags(CodeGenOpts->DwarfDebugFlags);
  if (Diags.ownsClient()) {
    Diags.setClient(
        new ChainedDiagnosticConsumer(Diags.takeClient(), std::move(Logger)));
  } else {
    Diags.setClient(
        new ChainedDiagnosticConsumer(Diags.getClient(), std::move(Logger)));
  }
}

static void SetupSerializedDiagnostics(DiagnosticOptions &DiagOpts,
                                       DiagnosticsEngine &Diags,
                                       StringRef OutputFile) {
  auto SerializedConsumer =
      clang::serialized_diags::create(OutputFile, DiagOpts);

  if (Diags.ownsClient()) {
    Diags.setClient(new ChainedDiagnosticConsumer(
        Diags.takeClient(), std::move(SerializedConsumer)));
  } else {
    Diags.setClient(new ChainedDiagnosticConsumer(
        Diags.getClient(), std::move(SerializedConsumer)));
  }
}

void CompilerInstance::createDiagnostics(llvm::vfs::FileSystem &VFS,
                                         DiagnosticConsumer *Client,
                                         bool ShouldOwnClient) {
  Diagnostics = createDiagnostics(VFS, getDiagnosticOpts(), Client,
                                  ShouldOwnClient, &getCodeGenOpts());
}

IntrusiveRefCntPtr<DiagnosticsEngine> CompilerInstance::createDiagnostics(
    llvm::vfs::FileSystem &VFS, DiagnosticOptions &Opts,
    DiagnosticConsumer *Client, bool ShouldOwnClient,
    const CodeGenOptions *CodeGenOpts) {
  auto Diags = llvm::makeIntrusiveRefCnt<DiagnosticsEngine>(
      DiagnosticIDs::create(), Opts);

  // Create the diagnostic client for reporting errors or for
  // implementing -verify.
  if (Client) {
    Diags->setClient(Client, ShouldOwnClient);
  } else if (Opts.getFormat() == DiagnosticOptions::SARIF) {
    Diags->setClient(new SARIFDiagnosticPrinter(llvm::errs(), Opts));
  } else
    Diags->setClient(new TextDiagnosticPrinter(llvm::errs(), Opts));

  // Chain in -verify checker, if requested.
  if (Opts.VerifyDiagnostics)
    Diags->setClient(new VerifyDiagnosticConsumer(*Diags));

  // Chain in -diagnostic-log-file dumper, if requested.
  if (!Opts.DiagnosticLogFile.empty())
    SetUpDiagnosticLog(Opts, CodeGenOpts, *Diags);

  if (!Opts.DiagnosticSerializationFile.empty())
    SetupSerializedDiagnostics(Opts, *Diags, Opts.DiagnosticSerializationFile);

  // Configure our handling of diagnostics.
  ProcessWarningOptions(*Diags, Opts, VFS);

  return Diags;
}

// File Manager

FileManager *CompilerInstance::createFileManager(
    IntrusiveRefCntPtr<llvm::vfs::FileSystem> VFS) {
  if (!VFS)
    VFS = FileMgr ? FileMgr->getVirtualFileSystemPtr()
                  : createVFSFromCompilerInvocation(getInvocation(),
                                                    getDiagnostics(), CAS);
  assert(VFS && "FileManager has no VFS?");
  if (getFrontendOpts().ShowStats)
    VFS =
        llvm::makeIntrusiveRefCnt<llvm::vfs::TracingFileSystem>(std::move(VFS));
  FileMgr = llvm::makeIntrusiveRefCnt<FileManager>(getFileSystemOpts(),
                                                   std::move(VFS));
  return FileMgr.get();
}

// Source Manager

void CompilerInstance::createSourceManager(FileManager &FileMgr) {
  SourceMgr =
      llvm::makeIntrusiveRefCnt<SourceManager>(getDiagnostics(), FileMgr);
}

// Initialize the remapping of files to alternative contents, e.g.,
// those specified through other files.
static void InitializeFileRemapping(DiagnosticsEngine &Diags,
                                    SourceManager &SourceMgr,
                                    FileManager &FileMgr,
                                    const PreprocessorOptions &InitOpts) {
  // Remap files in the source manager (with buffers).
  for (const auto &RB : InitOpts.RemappedFileBuffers) {
    // Create the file entry for the file that we're mapping from.
    FileEntryRef FromFile =
        FileMgr.getVirtualFileRef(RB.first, RB.second->getBufferSize(), 0);

    // Override the contents of the "from" file with the contents of the
    // "to" file. If the caller owns the buffers, then pass a MemoryBufferRef;
    // otherwise, pass as a std::unique_ptr<MemoryBuffer> to transfer ownership
    // to the SourceManager.
    if (InitOpts.RetainRemappedFileBuffers)
      SourceMgr.overrideFileContents(FromFile, RB.second->getMemBufferRef());
    else
      SourceMgr.overrideFileContents(
          FromFile, std::unique_ptr<llvm::MemoryBuffer>(RB.second));
  }

  // Remap files in the source manager (with other files).
  for (const auto &RF : InitOpts.RemappedFiles) {
    // Find the file that we're mapping to.
    OptionalFileEntryRef ToFile = FileMgr.getOptionalFileRef(RF.second);
    if (!ToFile) {
      Diags.Report(diag::err_fe_remap_missing_to_file) << RF.first << RF.second;
      continue;
    }

    // Create the file entry for the file that we're mapping from.
    FileEntryRef FromFile =
        FileMgr.getVirtualFileRef(RF.first, ToFile->getSize(), 0);

    // Override the contents of the "from" file with the contents of
    // the "to" file.
    SourceMgr.overrideFileContents(FromFile, *ToFile);
  }

  SourceMgr.setOverridenFilesKeepOriginalName(
      InitOpts.RemappedFilesKeepOriginalName);
}

// Preprocessor

void CompilerInstance::createPreprocessor(TranslationUnitKind TUKind) {
  const PreprocessorOptions &PPOpts = getPreprocessorOpts();

  // The AST reader holds a reference to the old preprocessor (if any).
  TheASTReader.reset();

  // Create the Preprocessor.
  HeaderSearch *HeaderInfo =
      new HeaderSearch(getHeaderSearchOpts(), getSourceManager(),
                       getDiagnostics(), getLangOpts(), &getTarget());
  PP = std::make_shared<Preprocessor>(Invocation->getPreprocessorOpts(),
                                      getDiagnostics(), getLangOpts(),
                                      getSourceManager(), *HeaderInfo, *this,
                                      /*IdentifierInfoLookup=*/nullptr,
                                      /*OwnsHeaderSearch=*/true, TUKind);
  getTarget().adjust(getDiagnostics(), getLangOpts(), getAuxTarget());
  PP->Initialize(getTarget(), getAuxTarget());

  if (PPOpts.DetailedRecord)
    PP->createPreprocessingRecord();

  // Apply remappings to the source manager.
  InitializeFileRemapping(PP->getDiagnostics(), PP->getSourceManager(),
                          PP->getFileManager(), PPOpts);

  // Predefine macros and configure the preprocessor.
  InitializePreprocessor(*PP, PPOpts, getPCHContainerReader(),
                         getFrontendOpts(), getCodeGenOpts());

  // Initialize the header search object.  In CUDA compilations, we use the aux
  // triple (the host triple) to initialize our header search, since we need to
  // find the host headers in order to compile the CUDA code.
  const llvm::Triple *HeaderSearchTriple = &PP->getTargetInfo().getTriple();
  if (PP->getTargetInfo().getTriple().getOS() == llvm::Triple::CUDA &&
      PP->getAuxTargetInfo())
    HeaderSearchTriple = &PP->getAuxTargetInfo()->getTriple();

  ApplyHeaderSearchOptions(PP->getHeaderSearchInfo(), getHeaderSearchOpts(),
                           PP->getLangOpts(), *HeaderSearchTriple);

  PP->setPreprocessedOutput(getPreprocessorOutputOpts().ShowCPP);

  if (PP->getLangOpts().Modules && PP->getLangOpts().ImplicitModules) {
    std::string ModuleHash = getInvocation().getModuleHash(getDiagnostics());
    PP->getHeaderSearchInfo().setModuleHash(ModuleHash);
    PP->getHeaderSearchInfo().setModuleCachePath(
        getSpecificModuleCachePath(ModuleHash));
  }

  // Handle generating dependencies, if requested.
  const DependencyOutputOptions &DepOpts = getDependencyOutputOpts();
  if (!DepOpts.OutputFile.empty())
    addDependencyCollector(
        std::make_shared<DependencyFileGenerator>(DepOpts, TheOutputBackend));
  if (!DepOpts.DOTOutputFile.empty())
    AttachDependencyGraphGen(*PP, DepOpts.DOTOutputFile,
                             getHeaderSearchOpts().Sysroot);

  // If we don't have a collector, but we are collecting module dependencies,
  // then we're the top level compiler instance and need to create one.
  if (!ModuleDepCollector && !DepOpts.ModuleDependencyOutputDir.empty()) {
    ModuleDepCollector = std::make_shared<ModuleDependencyCollector>(
        DepOpts.ModuleDependencyOutputDir);
  }

  // If there is a module dep collector, register with other dep collectors
  // and also (a) collect header maps and (b) TODO: input vfs overlay files.
  if (ModuleDepCollector) {
    addDependencyCollector(ModuleDepCollector);
    collectHeaderMaps(PP->getHeaderSearchInfo(), ModuleDepCollector);
    collectIncludePCH(*this, ModuleDepCollector);
    collectVFSEntries(*this, ModuleDepCollector);
  }

  // Modules need an output manager.
  if (!hasOutputBackend())
    createOutputBackend();

  for (auto &Listener : DependencyCollectors)
    Listener->attachToPreprocessor(*PP);

  // Handle generating header include information, if requested.
  if (DepOpts.ShowHeaderIncludes)
    AttachHeaderIncludeGen(*PP, DepOpts);
  if (!DepOpts.HeaderIncludeOutputFile.empty()) {
    StringRef OutputPath = DepOpts.HeaderIncludeOutputFile;
    if (OutputPath == "-")
      OutputPath = "";
    AttachHeaderIncludeGen(*PP, DepOpts,
                           /*ShowAllHeaders=*/true, OutputPath,
                           /*ShowDepth=*/false);
  }

  if (DepOpts.ShowIncludesDest != ShowIncludesDestination::None) {
    AttachHeaderIncludeGen(*PP, DepOpts,
                           /*ShowAllHeaders=*/true, /*OutputPath=*/"",
                           /*ShowDepth=*/true, /*MSStyle=*/true);
  }

  if (GetDependencyDirectives)
    PP->setDependencyDirectivesGetter(*GetDependencyDirectives);
}

std::string CompilerInstance::getSpecificModuleCachePath(StringRef ModuleHash) {
  // Set up the module path, including the hash for the module-creation options.
  SmallString<256> SpecificModuleCache(getHeaderSearchOpts().ModuleCachePath);
  if (!SpecificModuleCache.empty() && !getHeaderSearchOpts().DisableModuleHash)
    llvm::sys::path::append(SpecificModuleCache, ModuleHash);
  return std::string(SpecificModuleCache);
}

// ASTContext

void CompilerInstance::createASTContext() {
  Preprocessor &PP = getPreprocessor();
  auto Context = llvm::makeIntrusiveRefCnt<ASTContext>(
      getLangOpts(), PP.getSourceManager(), PP.getIdentifierTable(),
      PP.getSelectorTable(), PP.getBuiltinInfo(), PP.TUKind);
  Context->InitBuiltinTypes(getTarget(), getAuxTarget());
  setASTContext(std::move(Context));
}

// ExternalASTSource

namespace {
// Helper to recursively read the module names for all modules we're adding.
// We mark these as known and redirect any attempt to load that module to
// the files we were handed.
struct ReadModuleNames : ASTReaderListener {
  Preprocessor &PP;
  llvm::SmallVector<std::string, 8> LoadedModules;

  ReadModuleNames(Preprocessor &PP) : PP(PP) {}

  void ReadModuleName(StringRef ModuleName) override {
    // Keep the module name as a string for now. It's not safe to create a new
    // IdentifierInfo from an ASTReader callback.
    LoadedModules.push_back(ModuleName.str());
  }

  void registerAll() {
    ModuleMap &MM = PP.getHeaderSearchInfo().getModuleMap();
    for (const std::string &LoadedModule : LoadedModules)
      MM.cacheModuleLoad(*PP.getIdentifierInfo(LoadedModule),
                         MM.findOrLoadModule(LoadedModule));
    LoadedModules.clear();
  }

  void markAllUnavailable() {
    for (const std::string &LoadedModule : LoadedModules) {
      if (Module *M = PP.getHeaderSearchInfo().getModuleMap().findOrLoadModule(
              LoadedModule)) {
        M->HasIncompatibleModuleFile = true;

        // Mark module as available if the only reason it was unavailable
        // was missing headers.
        SmallVector<Module *, 2> Stack;
        Stack.push_back(M);
        while (!Stack.empty()) {
          Module *Current = Stack.pop_back_val();
          if (Current->IsUnimportable) continue;
          Current->IsAvailable = true;
          auto SubmodulesRange = Current->submodules();
          llvm::append_range(Stack, SubmodulesRange);
        }
      }
    }
    LoadedModules.clear();
  }
};

class CompileCacheASTReaderHelper : public ASTReaderListener {
public:
  CompileCacheASTReaderHelper(cas::ObjectStore &CAS, cas::ActionCache &Cache,
                              ModuleCache &ModCache, DiagnosticsEngine &Diags)
      : CAS(CAS), Cache(Cache), ModCache(ModCache), Diags(Diags) {}

  bool readCASFileSystemRootID(StringRef RootID, bool Complain) override;
  bool readIncludeTreeID(StringRef ID, bool Complain) override;
  bool readModuleCacheKey(StringRef ModuleName, StringRef Filename,
                          StringRef CacheKey) override;

private:
  bool checkCASID(bool Complain, StringRef RootID, unsigned ParseDiagID,
                  unsigned MissingDiagID);

  cas::ObjectStore &CAS;
  cas::ActionCache &Cache;
  ModuleCache &ModCache;
  DiagnosticsEngine &Diags;
};
} // namespace

void CompilerInstance::createPCHExternalASTSource(
    StringRef Path, DisableValidationForModuleKind DisableValidation,
    bool AllowPCHWithCompilerErrors, void *DeserializationListener,
    bool OwnDeserializationListener,
    std::unique_ptr<llvm::MemoryBuffer> PCHBuffer) {
  bool Preamble = getPreprocessorOpts().PrecompiledPreambleBytes.first != 0;
  TheASTReader = createPCHExternalASTSource(
      Path, getHeaderSearchOpts().Sysroot, DisableValidation,
      AllowPCHWithCompilerErrors, getPreprocessor(), getModuleCache(),
      getASTContext(), getPCHContainerReader(), getCodeGenOpts(),
      getFrontendOpts().ModuleFileExtensions, DependencyCollectors,
      DeserializationListener, OwnDeserializationListener, Preamble,
      getFrontendOpts().UseGlobalModuleIndex, getOrCreateObjectStore(),
      getOrCreateActionCache(), getFrontendOpts().ModuleLoadIgnoreCAS,
      std::move(PCHBuffer));
}

IntrusiveRefCntPtr<ASTReader> CompilerInstance::createPCHExternalASTSource(
    StringRef Path, StringRef Sysroot,
    DisableValidationForModuleKind DisableValidation,
    bool AllowPCHWithCompilerErrors, Preprocessor &PP, ModuleCache &ModCache,
    ASTContext &Context, const PCHContainerReader &PCHContainerRdr,
    const CodeGenOptions &CodeGenOpts,
    ArrayRef<std::shared_ptr<ModuleFileExtension>> Extensions,
    ArrayRef<std::shared_ptr<DependencyCollector>> DependencyCollectors,
    void *DeserializationListener, bool OwnDeserializationListener,
    bool Preamble, bool UseGlobalModuleIndex, cas::ObjectStore &CAS,
    cas::ActionCache &Cache, bool ignoreCAS,
    std::unique_ptr<llvm::MemoryBuffer> PCHBuffer) {
  const HeaderSearchOptions &HSOpts =
      PP.getHeaderSearchInfo().getHeaderSearchOpts();

  auto Reader = llvm::makeIntrusiveRefCnt<ASTReader>(
      PP, ModCache, &Context, PCHContainerRdr, CodeGenOpts, Extensions,
      Sysroot.empty() ? "" : Sysroot.data(), DisableValidation,
      AllowPCHWithCompilerErrors, /*AllowConfigurationMismatch*/ false,
      HSOpts.ModulesValidateSystemHeaders,
      HSOpts.ModulesForceValidateUserHeaders,
      HSOpts.ValidateASTInputFilesContent, UseGlobalModuleIndex);

  // We need the external source to be set up before we read the AST, because
  // eagerly-deserialized declarations may use it.
  Context.setExternalSource(Reader);

  Reader->setDeserializationListener(
      static_cast<ASTDeserializationListener *>(DeserializationListener),
      /*TakeOwnership=*/OwnDeserializationListener);

  for (auto &Listener : DependencyCollectors)
    Listener->attachToASTReader(*Reader);

  if (!ignoreCAS)
    Reader->addListener(std::make_unique<CompileCacheASTReaderHelper>(
        CAS, Cache, ModCache, PP.getDiagnostics()));

  auto Listener = std::make_unique<ReadModuleNames>(PP);
  auto &ListenerRef = *Listener;
  ASTReader::ListenerScope ReadModuleNamesListener(*Reader,
                                                   std::move(Listener));

  if (PCHBuffer) {
    Reader->addInMemoryBuffer(Path, std::move(PCHBuffer));
  }

  switch (Reader->ReadAST(Path,
                          Preamble ? serialization::MK_Preamble
                                   : serialization::MK_PCH,
                          SourceLocation(),
                          ASTReader::ARR_None)) {
  case ASTReader::Success:
    // Set the predefines buffer as suggested by the PCH reader. Typically, the
    // predefines buffer will be empty.
    PP.setPredefines(Reader->getSuggestedPredefines());
    ListenerRef.registerAll();
    return Reader;

  case ASTReader::Failure:
    // Unrecoverable failure: don't even try to process the input file.
    break;

  case ASTReader::Missing:
  case ASTReader::OutOfDate:
  case ASTReader::VersionMismatch:
  case ASTReader::ConfigurationMismatch:
  case ASTReader::HadErrors:
    // No suitable PCH file could be found. Return an error.
    break;
  }

  ListenerRef.markAllUnavailable();
  Context.setExternalSource(nullptr);
  return nullptr;
}

// Code Completion

static bool EnableCodeCompletion(Preprocessor &PP,
                                 StringRef Filename,
                                 unsigned Line,
                                 unsigned Column) {
  // Tell the source manager to chop off the given file at a specific
  // line and column.
  auto Entry = PP.getFileManager().getOptionalFileRef(Filename);
  if (!Entry) {
    PP.getDiagnostics().Report(diag::err_fe_invalid_code_complete_file)
      << Filename;
    return true;
  }

  // Truncate the named file at the given line/column.
  PP.SetCodeCompletionPoint(*Entry, Line, Column);
  return false;
}

void CompilerInstance::createCodeCompletionConsumer() {
  const ParsedSourceLocation &Loc = getFrontendOpts().CodeCompletionAt;
  if (!CompletionConsumer) {
    setCodeCompletionConsumer(createCodeCompletionConsumer(
        getPreprocessor(), Loc.FileName, Loc.Line, Loc.Column,
        getFrontendOpts().CodeCompleteOpts, llvm::outs()));
    return;
  } else if (EnableCodeCompletion(getPreprocessor(), Loc.FileName,
                                  Loc.Line, Loc.Column)) {
    setCodeCompletionConsumer(nullptr);
    return;
  }
}

void CompilerInstance::createFrontendTimer() {
  timerGroup.reset(new llvm::TimerGroup("clang", "Clang time report"));
  FrontendTimer.reset(new llvm::Timer("frontend", "Front end", *timerGroup));
}

CodeCompleteConsumer *
CompilerInstance::createCodeCompletionConsumer(Preprocessor &PP,
                                               StringRef Filename,
                                               unsigned Line,
                                               unsigned Column,
                                               const CodeCompleteOptions &Opts,
                                               raw_ostream &OS) {
  if (EnableCodeCompletion(PP, Filename, Line, Column))
    return nullptr;

  // Set up the creation routine for code-completion.
  return new PrintingCodeCompleteConsumer(Opts, OS);
}

static void loadAPINotesFromIncludeTree(cas::ObjectStore &DB,
                                        api_notes::APINotesManager &APINotes,
                                        DiagnosticsEngine &Diags,
                                        StringRef IncludeTreeRootID) {
  Expected<llvm::cas::CASID> RootID = DB.parseID(IncludeTreeRootID);
  if (!RootID) {
    llvm::consumeError(RootID.takeError());
    Diags.Report(diag::err_cas_cannot_parse_include_tree_id)
        << IncludeTreeRootID;
    return;
  }
  std::optional<llvm::cas::ObjectRef> Ref = DB.getReference(*RootID);
  if (!Ref) {
    Diags.Report(diag::err_cas_missing_include_tree_id) << IncludeTreeRootID;
    return;
  }
  auto Root = cas::IncludeTreeRoot::get(DB, *Ref);
  if (!Root) {
    consumeError(Root.takeError());
    Diags.Report(diag::err_cas_missing_include_tree_id) << IncludeTreeRootID;
    return;
  }
  auto Notes = Root->getAPINotes();
  if (!Notes) {
    consumeError(Notes.takeError());
    Diags.Report(diag::err_cas_cannot_load_api_notes_include_tree)
        << IncludeTreeRootID;
    return;
  }
  if (!*Notes)
    return;
  std::vector<StringRef> Buffers;

  if (auto E = (*Notes)->forEachAPINotes([&](StringRef Buffer) {
        Buffers.push_back(Buffer);
        return llvm::Error::success();
      })) {
    consumeError(std::move(E));
    Diags.Report(diag::err_cas_cannot_load_api_notes_include_tree)
        << IncludeTreeRootID;
    return;
  }

  APINotes.loadCurrentModuleAPINotesFromBuffer(Buffers);
}

void CompilerInstance::createSema(TranslationUnitKind TUKind,
                                  CodeCompleteConsumer *CompletionConsumer) {
  TheSema.reset(new Sema(getPreprocessor(), getASTContext(), getASTConsumer(),
                         TUKind, CompletionConsumer));

  // Set up API notes.
  TheSema->APINotes.setSwiftVersion(getAPINotesOpts().SwiftVersion);

  // If we're building a module and are supposed to load API notes,
  // notify the API notes manager.
  if (auto currentModule = getPreprocessor().getCurrentModule()) {
    // If using include tree, APINotes for current module is loaded from include
    // tree.
    if (getFrontendOpts().CASIncludeTreeID.empty())
      (void)TheSema->APINotes.loadCurrentModuleAPINotes(
          currentModule, getLangOpts().APINotesModules,
          getAPINotesOpts().ModuleSearchPaths);
    else
      loadAPINotesFromIncludeTree(
          *getCASOpts().getOrCreateDatabases(getDiagnostics()).first,
          TheSema->APINotes, getDiagnostics(),
          getFrontendOpts().CASIncludeTreeID);

    // Check for any attributes we should add to the module
    for (auto reader : TheSema->APINotes.getCurrentModuleReaders()) {
      // swift_infer_import_as_member
      if (reader->getModuleOptions().SwiftInferImportAsMember) {
        currentModule->IsSwiftInferImportAsMember = true;
        break;
      }
    }
  }

  // Attach the external sema source if there is any.
  if (ExternalSemaSrc) {
    TheSema->addExternalSource(ExternalSemaSrc);
    ExternalSemaSrc->InitializeSema(*TheSema);
  }
}

// Output Files

void CompilerInstance::clearOutputFiles(bool EraseFiles) {
  // The ASTConsumer can own streams that write to the output files.
  assert(!hasASTConsumer() && "ASTConsumer should be reset");
  if (!EraseFiles) {
    for (auto &O : OutputFiles)
      llvm::handleAllErrors(
          O.keep(),
          [&](const llvm::vfs::TempFileOutputError &E) {
            getDiagnostics().Report(diag::err_unable_to_rename_temp)
                << E.getTempPath() << E.getOutputPath()
                << E.convertToErrorCode().message();
          },
          [&](const llvm::vfs::OutputError &E) {
            getDiagnostics().Report(diag::err_fe_unable_to_open_output)
                << E.getOutputPath() << E.convertToErrorCode().message();
          },
          [&](const llvm::ErrorInfoBase &EIB) { // Handle any remaining error
            getDiagnostics().Report(diag::err_fe_unable_to_open_output)
                << O.getPath() << EIB.message();
          });
  }
  OutputFiles.clear();
  if (DeleteBuiltModules) {
    for (auto &Module : BuiltModules)
      llvm::sys::fs::remove(Module.second);
    BuiltModules.clear();
  }
}

std::unique_ptr<raw_pwrite_stream> CompilerInstance::createDefaultOutputFile(
    bool Binary, StringRef InFile, StringRef Extension, bool RemoveFileOnSignal,
    bool CreateMissingDirectories, bool ForceUseTemporary) {
  StringRef OutputPath = getFrontendOpts().OutputFile;
  std::optional<SmallString<128>> PathStorage;
  if (OutputPath.empty()) {
    if (InFile == "-" || Extension.empty()) {
      OutputPath = "-";
    } else {
      PathStorage.emplace(InFile);
      llvm::sys::path::replace_extension(*PathStorage, Extension);
      OutputPath = *PathStorage;
    }
  }

  return createOutputFile(OutputPath, Binary, RemoveFileOnSignal,
                          getFrontendOpts().UseTemporary || ForceUseTemporary,
                          CreateMissingDirectories);
}

std::unique_ptr<raw_pwrite_stream> CompilerInstance::createNullOutputFile() {
  return std::make_unique<llvm::raw_null_ostream>();
}

void CompilerInstance::setOutputBackend(
    IntrusiveRefCntPtr<llvm::vfs::OutputBackend> NewOutputs) {
  assert(!TheOutputBackend && "Already has an output manager");
  TheOutputBackend = std::move(NewOutputs);
}

void CompilerInstance::createOutputBackend() {
  assert(!TheOutputBackend && "Already has an output manager");
  TheOutputBackend = llvm::makeIntrusiveRefCnt<llvm::vfs::OnDiskOutputBackend>();
}

llvm::vfs::OutputBackend &CompilerInstance::getOutputBackend() {
  assert(TheOutputBackend);
  return *TheOutputBackend;
}

llvm::vfs::OutputBackend &CompilerInstance::getOrCreateOutputBackend() {
  if (!hasOutputBackend())
    createOutputBackend();
  return getOutputBackend();
}

std::pair<std::shared_ptr<llvm::cas::ObjectStore>,
          std::shared_ptr<llvm::cas::ActionCache>>
CompilerInstance::createCASDatabases() {
  // Create a new CAS databases from the CompilerInvocation. Future calls to
  // createFileManager() will use the same CAS.
  std::tie(CAS, ActionCache) =
      getInvocation().getCASOpts().getOrCreateDatabases(
          getDiagnostics(),
          /*CreateEmptyCASOnFailure=*/true);
  return {CAS, ActionCache};
}

llvm::cas::ObjectStore &CompilerInstance::getOrCreateObjectStore() {
  if (!CAS)
    createCASDatabases();
  return *CAS;
}

llvm::cas::ActionCache &CompilerInstance::getOrCreateActionCache() {
  if (!ActionCache)
    createCASDatabases();
  return *ActionCache;
}

std::unique_ptr<raw_pwrite_stream>
CompilerInstance::createOutputFile(StringRef OutputPath, bool Binary,
                                   bool RemoveFileOnSignal, bool UseTemporary,
                                   bool CreateMissingDirectories) {
  Expected<std::unique_ptr<raw_pwrite_stream>> OS =
      createOutputFileImpl(OutputPath, Binary, RemoveFileOnSignal, UseTemporary,
                           CreateMissingDirectories);
  if (OS)
    return std::move(*OS);
  getDiagnostics().Report(diag::err_fe_unable_to_open_output)
      << OutputPath << errorToErrorCode(OS.takeError()).message();
  return nullptr;
}

void CompilerInstance::initializeDelayedInputFileFromCAS() {
  auto &Opts = Invocation->getFrontendOpts();
  // Return if no need to initialize or already initialized.
  if ((Opts.CASIncludeTreeID.empty() && Opts.CASInputFileCASID.empty()) ||
      !Opts.Inputs.empty())
    return;

  assert(hasDiagnostics() && "need diagnostics engine for CAS loading");

  // If there is include tree, initialize the inputs from CAS.
  auto reportError = [&](llvm::Error &&E) {
    Diagnostics->Report(diag::err_fe_unable_to_load_include_tree)
        << Opts.CASIncludeTreeID << llvm::toString(std::move(E));
  };
  auto CAS = Invocation->getCASOpts().getOrCreateDatabases(*Diagnostics).first;
  if (!CAS)
    return;
  if (!Opts.CASIncludeTreeID.empty()) {
    auto ID = CAS->parseID(Opts.CASIncludeTreeID);
    if (!ID)
      return reportError(ID.takeError());
    auto Object = CAS->getReference(*ID);
    if (!Object)
      return reportError(llvm::cas::ObjectStore::createUnknownObjectError(*ID));
    auto Root = cas::IncludeTreeRoot::get(*CAS, *Object);
    if (!Root)
      return reportError(Root.takeError());
    auto MainTree = Root->getMainFileTree();
    if (!MainTree)
      return reportError(MainTree.takeError());
    auto BaseFile = MainTree->getBaseFile();
    if (!BaseFile)
      return reportError(BaseFile.takeError());
    auto FilenameBlob = BaseFile->getFilename();
    if (!FilenameBlob)
      return reportError(FilenameBlob.takeError());

    auto InputFilename = FilenameBlob->getData();

    if (InputFilename != Module::getModuleInputBufferName()) {
      Opts.Inputs.emplace_back(Root->getRef(), InputFilename, Opts.DashX,
                               /*isSystem=*/false);
    } else {
      assert(Opts.ProgramAction == frontend::GenerateModule);

      auto Kind = Opts.DashX.withFormat(InputKind::Source);
      auto Contents = BaseFile->getContents();
      if (!Contents)
        return reportError(Contents.takeError());
      auto Buffer = llvm::MemoryBufferRef(Contents->getData(), InputFilename);
      Opts.Inputs.emplace_back(Root->getRef(), Buffer, Kind,
                               (bool)Opts.IsSystemModule);
    }
    return;
  }
  if (!Opts.CASInputFileCASID.empty()) {
    auto ID = CAS->parseID(Opts.CASInputFileCASID);
    if (!ID)
      return reportError(ID.takeError());
    auto ValueRef = CAS->getReference(*ID);
    if (!ValueRef)
      return reportError(llvm::cas::ObjectStore::createUnknownObjectError(*ID));

    cas::CompileJobResultSchema Schema(*CAS);
    auto Result = Schema.load(*ValueRef);
    if (!Result)
      return reportError(Result.takeError());
    auto Output =
        Result->getOutput(cas::CompileJobCacheResult::OutputKind::MainOutput);
    if (!Output)
      return reportError(
          llvm::createStringError("unable to get the main compilation output"));

    auto OutProxy = CAS->getProxy(Output->Object);
    if (!OutProxy)
      return reportError(OutProxy.takeError());

    auto Buff = llvm::MemoryBufferRef(OutProxy->getData(), "<input>");
    Opts.Inputs.emplace_back(Buff, Opts.DashX, /*IsSystem=*/false);
    return;
  }
}

Expected<std::unique_ptr<llvm::raw_pwrite_stream>>
CompilerInstance::createOutputFileImpl(StringRef OutputPath, bool Binary,
                                       bool RemoveFileOnSignal,
                                       bool UseTemporary,
                                       bool CreateMissingDirectories) {
  assert((!CreateMissingDirectories || UseTemporary) &&
         "CreateMissingDirectories is only allowed when using temporary files");

  // If '-working-directory' was passed, the output filename should be
  // relative to that.
  std::optional<SmallString<128>> AbsPath;
  if (OutputPath != "-" && !llvm::sys::path::is_absolute(OutputPath)) {
    assert(hasFileManager() &&
           "File Manager is required to fix up relative path.\n");

    AbsPath.emplace(OutputPath);
    FileMgr->FixupRelativePath(*AbsPath);
    OutputPath = *AbsPath;
  }

  using namespace llvm::vfs;
  Expected<OutputFile> O = getOrCreateOutputBackend().createFile(
      OutputPath,
      OutputConfig()
          .setTextWithCRLF(!Binary)
          .setDiscardOnSignal(RemoveFileOnSignal)
          .setAtomicWrite(UseTemporary)
          .setImplyCreateDirectories(UseTemporary && CreateMissingDirectories));
  if (!O)
    return O.takeError();

  O->discardOnDestroy([](llvm::Error E) { consumeError(std::move(E)); });
  OutputFiles.push_back(std::move(*O));
  return OutputFiles.back().createProxy();
}

// Initialization Utilities

bool CompilerInstance::InitializeSourceManager(const FrontendInputFile &Input){
  return InitializeSourceManager(Input, getDiagnostics(), getFileManager(),
                                 getSourceManager());
}

// static
bool CompilerInstance::InitializeSourceManager(const FrontendInputFile &Input,
                                               DiagnosticsEngine &Diags,
                                               FileManager &FileMgr,
                                               SourceManager &SourceMgr) {
  SrcMgr::CharacteristicKind Kind =
      Input.getKind().getFormat() == InputKind::ModuleMap
          ? Input.isSystem() ? SrcMgr::C_System_ModuleMap
                             : SrcMgr::C_User_ModuleMap
          : Input.isSystem() ? SrcMgr::C_System : SrcMgr::C_User;

  if (Input.isBuffer()) {
    SourceMgr.setMainFileID(SourceMgr.createFileID(Input.getBuffer(), Kind));
    assert(SourceMgr.getMainFileID().isValid() &&
           "Couldn't establish MainFileID!");
    return true;
  }

  StringRef InputFile = Input.getFile();

  // Figure out where to get and map in the main file.
  auto FileOrErr = InputFile == "-"
                       ? FileMgr.getSTDIN()
                       : FileMgr.getFileRef(InputFile, /*OpenFile=*/true);
  if (!FileOrErr) {
    auto EC = llvm::errorToErrorCode(FileOrErr.takeError());
    if (InputFile != "-")
      Diags.Report(diag::err_fe_error_reading) << InputFile << EC.message();
    else
      Diags.Report(diag::err_fe_error_reading_stdin) << EC.message();
    return false;
  }

  SourceMgr.setMainFileID(
      SourceMgr.createFileID(*FileOrErr, SourceLocation(), Kind));

  assert(SourceMgr.getMainFileID().isValid() &&
         "Couldn't establish MainFileID!");
  return true;
}

// High-Level Operations

bool CompilerInstance::ExecuteAction(FrontendAction &Act) {
  assert(hasDiagnostics() && "Diagnostics engine is not initialized!");
  assert(!getFrontendOpts().ShowHelp && "Client must handle '-help'!");
  assert(!getFrontendOpts().ShowVersion && "Client must handle '-version'!");

  // Mark this point as the bottom of the stack if we don't have somewhere
  // better. We generally expect frontend actions to be invoked with (nearly)
  // DesiredStackSpace available.
  noteBottomOfStack();

  auto FinishDiagnosticClient = llvm::make_scope_exit([&]() {
    if (!getFrontendOpts().MayEmitDiagnosticsAfterProcessingSourceFiles) {
      // Notify the diagnostic client that all files were processed.
      getDiagnosticClient().finish();
    }
  });

  raw_ostream &OS = getVerboseOutputStream();

  if (!Act.PrepareToExecute(*this))
    return false;

  if (!createTarget())
    return false;

  // rewriter project will change target built-in bool type from its default.
  if (getFrontendOpts().ProgramAction == frontend::RewriteObjC)
    getTarget().noSignedCharForObjCBool();

  // Validate/process some options.
  if (getHeaderSearchOpts().Verbose)
    OS << "clang -cc1 version " CLANG_VERSION_STRING << " based upon LLVM "
       << LLVM_VERSION_STRING << " default target "
       << llvm::sys::getDefaultTargetTriple() << "\n";

  if (getFrontendOpts().ShowStats || !getFrontendOpts().StatsFile.empty())
    llvm::EnableStatistics(false);

  // Sort vectors containing toc data and no toc data variables to facilitate
  // binary search later.
  llvm::sort(getCodeGenOpts().TocDataVarsUserSpecified);
  llvm::sort(getCodeGenOpts().NoTocDataVars);

  initializeDelayedInputFileFromCAS();

  for (const FrontendInputFile &FIF : getFrontendOpts().Inputs) {
    // Reset the ID tables if we are reusing the SourceManager and parsing
    // regular files.
    if (hasSourceManager() && !Act.isModelParsingAction())
      getSourceManager().clearIDTables();

    if (Act.BeginSourceFile(*this, FIF)) {
      if (llvm::Error Err = Act.Execute()) {
        consumeError(std::move(Err)); // FIXME this drops errors on the floor.
      }
      Act.EndSourceFile();
    }
  }

  printDiagnosticStats();

  if (getFrontendOpts().ShowStats) {
    if (hasFileManager()) {
      getFileManager().PrintStats();
      OS << '\n';
    }
    llvm::PrintStatistics(OS);
  }
  StringRef StatsFile = getFrontendOpts().StatsFile;
  if (!StatsFile.empty()) {
    llvm::sys::fs::OpenFlags FileFlags = llvm::sys::fs::OF_TextWithCRLF;
    if (getFrontendOpts().AppendStats)
      FileFlags |= llvm::sys::fs::OF_Append;
    std::error_code EC;
    auto StatS =
        std::make_unique<llvm::raw_fd_ostream>(StatsFile, EC, FileFlags);
    if (EC) {
      getDiagnostics().Report(diag::warn_fe_unable_to_open_stats_file)
          << StatsFile << EC.message();
    } else {
      llvm::PrintStatisticsJSON(*StatS);
    }
  }

  return !getDiagnostics().getClient()->getNumErrors();
}

void CompilerInstance::printDiagnosticStats() {
  if (!getDiagnosticOpts().ShowCarets)
    return;

  raw_ostream &OS = getVerboseOutputStream();

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
  if (NumWarnings || NumErrors) {
    OS << " generated";
    if (getLangOpts().CUDA) {
      if (!getLangOpts().CUDAIsDevice) {
        OS << " when compiling for host";
      } else {
        OS << " when compiling for " << getTargetOpts().CPU;
      }
    }
    OS << ".\n";
  }
}

void CompilerInstance::LoadRequestedPlugins() {
  // Load any requested plugins.
  for (const std::string &Path : getFrontendOpts().Plugins) {
    std::string Error;
    if (llvm::sys::DynamicLibrary::LoadLibraryPermanently(Path.c_str(), &Error))
      getDiagnostics().Report(diag::err_fe_unable_to_load_plugin)
          << Path << Error;
  }

  // Check if any of the loaded plugins replaces the main AST action
  for (const FrontendPluginRegistry::entry &Plugin :
       FrontendPluginRegistry::entries()) {
    std::unique_ptr<PluginASTAction> P(Plugin.instantiate());
    if (P->getActionType() == PluginASTAction::ReplaceAction) {
      getFrontendOpts().ProgramAction = clang::frontend::PluginAction;
      getFrontendOpts().ActionName = Plugin.getName().str();
      break;
    }
  }
}

/// Determine the appropriate source input kind based on language
/// options.
static Language getLanguageFromOptions(const LangOptions &LangOpts) {
  if (LangOpts.OpenCL)
    return Language::OpenCL;
  if (LangOpts.CUDA)
    return Language::CUDA;
  if (LangOpts.ObjC)
    return LangOpts.CPlusPlus ? Language::ObjCXX : Language::ObjC;
  return LangOpts.CPlusPlus ? Language::CXX : Language::C;
}

std::unique_ptr<CompilerInstance> CompilerInstance::cloneForModuleCompileImpl(
    SourceLocation ImportLoc, StringRef ModuleName, FrontendInputFile Input,
    StringRef OriginalModuleMapFile, StringRef ModuleFileName,
    std::optional<ThreadSafeCloneConfig> ThreadSafeConfig) {
  // Construct a compiler invocation for creating this module.
  auto Invocation = std::make_shared<CompilerInvocation>(getInvocation());

  PreprocessorOptions &PPOpts = Invocation->getPreprocessorOpts();

  // For any options that aren't intended to affect how a module is built,
  // reset them to their default values.
  Invocation->resetNonModularOptions();

  // Remove any macro definitions that are explicitly ignored by the module.
  // They aren't supposed to affect how the module is built anyway.
  HeaderSearchOptions &HSOpts = Invocation->getHeaderSearchOpts();
  llvm::erase_if(PPOpts.Macros,
                 [&HSOpts](const std::pair<std::string, bool> &def) {
                   StringRef MacroDef = def.first;
                   return HSOpts.ModulesIgnoreMacros.contains(
                       llvm::CachedHashString(MacroDef.split('=').first));
                 });

  // If the original compiler invocation had -fmodule-name, pass it through.
  Invocation->getLangOpts().ModuleName =
      getInvocation().getLangOpts().ModuleName;

  // Note the name of the module we're building.
  Invocation->getLangOpts().CurrentModule = std::string(ModuleName);

  // If there is a module map file, build the module using the module map.
  // Set up the inputs/outputs so that we build the module from its umbrella
  // header.
  FrontendOptions &FrontendOpts = Invocation->getFrontendOpts();
  FrontendOpts.OutputFile = ModuleFileName.str();
  FrontendOpts.DisableFree = false;
  FrontendOpts.GenerateGlobalModuleIndex = false;
  FrontendOpts.BuildingImplicitModule = true;
  FrontendOpts.OriginalModuleMap = std::string(OriginalModuleMapFile);
  // Force implicitly-built modules to hash the content of the module file.
  HSOpts.ModulesHashContent = true;
  FrontendOpts.Inputs = {std::move(Input)};
  FrontendOpts.MayEmitDiagnosticsAfterProcessingSourceFiles = false;
  // Clear `IndexUnitOutputPath`. Otherwise the unit for the pcm will be named
  // after the primary source file, and will be overwritten when that file's
  // unit is finally written.
  FrontendOpts.IndexUnitOutputPath = "";
  if (FrontendOpts.IndexIgnorePcms) {
    // If we shouldn't index pcms, disable indexing by clearing the index store
    // path.
    FrontendOpts.IndexStorePath = "";
  }

  // Don't free the remapped file buffers; they are owned by our caller.
  PPOpts.RetainRemappedFileBuffers = true;

  DiagnosticOptions &DiagOpts = Invocation->getDiagnosticOpts();

  DiagOpts.VerifyDiagnostics = 0;
  assert(getInvocation().getModuleHash(getDiagnostics()) ==
         Invocation->getModuleHash(getDiagnostics()) &&
         "Module hash mismatch!");

  // Construct a compiler instance that will be used to actually create the
  // module.  Since we're sharing an in-memory module cache,
  // CompilerInstance::CompilerInstance is responsible for finalizing the
  // buffers to prevent use-after-frees.
  auto InstancePtr = std::make_unique<CompilerInstance>(
      std::move(Invocation), getPCHContainerOperations(), &getModuleCache());
  auto &Instance = *InstancePtr;

  auto &Inv = Instance.getInvocation();

  if (ThreadSafeConfig) {
    Instance.createFileManager(ThreadSafeConfig->getVFS());
  } else if (FrontendOpts.ModulesShareFileManager) {
    Instance.setFileManager(getFileManagerPtr());
  } else {
    Instance.createFileManager(getVirtualFileSystemPtr());
  }

  if (ThreadSafeConfig) {
    Instance.createDiagnostics(Instance.getVirtualFileSystem(),
                               &ThreadSafeConfig->getDiagConsumer(),
                               /*ShouldOwnClient=*/false);
  } else {
    Instance.createDiagnostics(
        Instance.getVirtualFileSystem(),
        new ForwardingDiagnosticConsumer(getDiagnosticClient()),
        /*ShouldOwnClient=*/true);
  }
  if (llvm::is_contained(DiagOpts.SystemHeaderWarningsModules, ModuleName))
    Instance.getDiagnostics().setSuppressSystemWarnings(false);

  Instance.createSourceManager(Instance.getFileManager());
  SourceManager &SourceMgr = Instance.getSourceManager();

  if (ThreadSafeConfig) {
    // Detecting cycles in the module graph is responsibility of the client.
  } else {
    // Note that this module is part of the module build stack, so that we
    // can detect cycles in the module graph.
    SourceMgr.setModuleBuildStack(getSourceManager().getModuleBuildStack());
    SourceMgr.pushModuleBuildStack(
        ModuleName, FullSourceLoc(ImportLoc, getSourceManager()));
  }

  // Make a copy for the new instance.
  Instance.FailedModules = FailedModules;

  if (GetDependencyDirectives)
    Instance.GetDependencyDirectives =
        GetDependencyDirectives->cloneFor(Instance.getFileManager());

  // Pass along the GenModuleActionWrapper callback
  auto WrapGenModuleAction = getGenModuleActionWrapper();
  Instance.setGenModuleActionWrapper(WrapGenModuleAction);

  assert(hasOutputBackend() &&
         "Expected an output manager to already be set up");
  if (ThreadSafeConfig) {
    // Create a clone of the existing output (pointing to the same destination).
    Instance.setOutputBackend(getOutputBackend().clone());
  } else {
    // Share the existing output manager.
    Instance.setOutputBackend(&getOutputBackend());
  }

  if (ThreadSafeConfig) {
    Instance.setModuleDepCollector(ThreadSafeConfig->getModuleDepCollector());
  } else {
    // If we're collecting module dependencies, we need to share a collector
    // between all of the module CompilerInstances. Other than that, we don't
    // want to produce any dependency output from the module build.
    Instance.setModuleDepCollector(getModuleDepCollector());
  }
  Inv.getDependencyOutputOpts() = DependencyOutputOptions();

  return InstancePtr;
}

bool CompilerInstance::compileModule(SourceLocation ImportLoc,
                                     StringRef ModuleName,
                                     StringRef ModuleFileName,
                                     CompilerInstance &Instance) {
  llvm::TimeTraceScope TimeScope("Module Compile", ModuleName);

  // Never compile a module that's already finalized - this would cause the
  // existing module to be freed, causing crashes if it is later referenced
  if (getModuleCache().getInMemoryModuleCache().isPCMFinal(ModuleFileName)) {
    getDiagnostics().Report(ImportLoc, diag::err_module_rebuild_finalized)
        << ModuleName;
    return false;
  }

  getDiagnostics().Report(ImportLoc, diag::remark_module_build)
      << ModuleName << ModuleFileName;

  // Execute the action to actually build the module in-place. Use a separate
  // thread so that we get a stack large enough.
  bool Crashed = !llvm::CrashRecoveryContext().RunSafelyOnNewStack(
      [&]() {
        std::unique_ptr<FrontendAction> Action(
            new GenerateModuleFromModuleMapAction);
        if (auto WrapGenModuleAction = Instance.getGenModuleActionWrapper())
          Action = WrapGenModuleAction(Instance.getFrontendOpts(),
                                       std::move(Action));
        Instance.ExecuteAction(*Action);
      },
      DesiredStackSize);

  getDiagnostics().Report(ImportLoc, diag::remark_module_build_done)
      << ModuleName;

  // Propagate the statistics to the parent FileManager.
  if (!getFrontendOpts().ModulesShareFileManager)
    getFileManager().AddStats(Instance.getFileManager());

  // Propagate the failed modules to the parent instance.
  FailedModules = std::move(Instance.FailedModules);

  if (Crashed) {
    // Clear the ASTConsumer if it hasn't been already, in case it owns streams
    // that must be closed before clearing output files.
    Instance.setSema(nullptr);
    Instance.setASTConsumer(nullptr);

    // Delete any remaining temporary files related to Instance.
    Instance.clearOutputFiles(/*EraseFiles=*/true);
  }

  // We've rebuilt a module. If we're allowed to generate or update the global
  // module index, record that fact in the importing compiler instance.
  if (getFrontendOpts().GenerateGlobalModuleIndex) {
    setBuildGlobalModuleIndex(true);
  }

  // If \p AllowPCMWithCompilerErrors is set return 'success' even if errors
  // occurred.
  return !Instance.getDiagnostics().hasErrorOccurred() ||
         Instance.getFrontendOpts().AllowPCMWithCompilerErrors;
}

static OptionalFileEntryRef getPublicModuleMap(FileEntryRef File,
                                               FileManager &FileMgr) {
  StringRef Filename = llvm::sys::path::filename(File.getName());
  SmallString<128> PublicFilename(File.getDir().getName());
  if (Filename == "module_private.map")
    llvm::sys::path::append(PublicFilename, "module.map");
  else if (Filename == "module.private.modulemap")
    llvm::sys::path::append(PublicFilename, "module.modulemap");
  else
    return std::nullopt;
  return FileMgr.getOptionalFileRef(PublicFilename);
}

std::unique_ptr<CompilerInstance> CompilerInstance::cloneForModuleCompile(
    SourceLocation ImportLoc, Module *Module, StringRef ModuleFileName,
    std::optional<ThreadSafeCloneConfig> ThreadSafeConfig) {
  StringRef ModuleName = Module->getTopLevelModuleName();

  InputKind IK(getLanguageFromOptions(getLangOpts()), InputKind::ModuleMap);

  // Get or create the module map that we'll use to build this module.
  ModuleMap &ModMap = getPreprocessor().getHeaderSearchInfo().getModuleMap();
  SourceManager &SourceMgr = getSourceManager();

  if (FileID ModuleMapFID = ModMap.getContainingModuleMapFileID(Module);
      ModuleMapFID.isValid()) {
    // We want to use the top-level module map. If we don't, the compiling
    // instance may think the containing module map is a top-level one, while
    // the importing instance knows it's included from a parent module map via
    // the extern directive. This mismatch could bite us later.
    SourceLocation Loc = SourceMgr.getIncludeLoc(ModuleMapFID);
    while (Loc.isValid() && isModuleMap(SourceMgr.getFileCharacteristic(Loc))) {
      ModuleMapFID = SourceMgr.getFileID(Loc);
      Loc = SourceMgr.getIncludeLoc(ModuleMapFID);
    }

    OptionalFileEntryRef ModuleMapFile =
        SourceMgr.getFileEntryRefForID(ModuleMapFID);
    assert(ModuleMapFile && "Top-level module map with no FileID");

    // Canonicalize compilation to start with the public module map. This is
    // vital for submodules declarations in the private module maps to be
    // correctly parsed when depending on a top level module in the public one.
    if (OptionalFileEntryRef PublicMMFile =
            getPublicModuleMap(*ModuleMapFile, getFileManager()))
      ModuleMapFile = PublicMMFile;

    StringRef ModuleMapFilePath = ModuleMapFile->getNameAsRequested();

    // Use the systemness of the module map as parsed instead of using the
    // IsSystem attribute of the module. If the module has [system] but the
    // module map is not in a system path, then this would incorrectly parse
    // any other modules in that module map as system too.
    const SrcMgr::SLocEntry &SLoc = SourceMgr.getSLocEntry(ModuleMapFID);
    bool IsSystem = isSystem(SLoc.getFile().getFileCharacteristic());

    // Use the module map where this module resides.
    return cloneForModuleCompileImpl(
        ImportLoc, ModuleName,
        FrontendInputFile(ModuleMapFilePath, IK, IsSystem),
        ModMap.getModuleMapFileForUniquing(Module)->getName(), ModuleFileName,
        std::move(ThreadSafeConfig));
  }

  // FIXME: We only need to fake up an input file here as a way of
  // transporting the module's directory to the module map parser. We should
  // be able to do that more directly, and parse from a memory buffer without
  // inventing this file.
  SmallString<128> FakeModuleMapFile(Module->Directory->getName());
  llvm::sys::path::append(FakeModuleMapFile, "__inferred_module.map");

  std::string InferredModuleMapContent;
  llvm::raw_string_ostream OS(InferredModuleMapContent);
  Module->print(OS);

  auto Instance = cloneForModuleCompileImpl(
      ImportLoc, ModuleName,
      FrontendInputFile(FakeModuleMapFile, IK, +Module->IsSystem),
      ModMap.getModuleMapFileForUniquing(Module)->getName(), ModuleFileName,
      std::move(ThreadSafeConfig));

  std::unique_ptr<llvm::MemoryBuffer> ModuleMapBuffer =
      llvm::MemoryBuffer::getMemBufferCopy(InferredModuleMapContent);
  FileEntryRef ModuleMapFile = Instance->getFileManager().getVirtualFileRef(
      FakeModuleMapFile, InferredModuleMapContent.size(), 0);
  Instance->getSourceManager().overrideFileContents(ModuleMapFile,
                                                    std::move(ModuleMapBuffer));

  return Instance;
}

/// Read the AST right after compiling the module.
static bool readASTAfterCompileModule(CompilerInstance &ImportingInstance,
                                      SourceLocation ImportLoc,
                                      SourceLocation ModuleNameLoc,
                                      Module *Module, StringRef ModuleFileName,
                                      bool *OutOfDate, bool *Missing) {
  DiagnosticsEngine &Diags = ImportingInstance.getDiagnostics();

  unsigned ModuleLoadCapabilities = ASTReader::ARR_Missing;
  if (OutOfDate)
    ModuleLoadCapabilities |= ASTReader::ARR_OutOfDate;

  // Try to read the module file, now that we've compiled it.
  ASTReader::ASTReadResult ReadResult =
      ImportingInstance.getASTReader()->ReadAST(
          ModuleFileName, serialization::MK_ImplicitModule, ImportLoc,
          ModuleLoadCapabilities);
  if (ReadResult == ASTReader::Success)
    return true;

  // The caller wants to handle out-of-date failures.
  if (OutOfDate && ReadResult == ASTReader::OutOfDate) {
    *OutOfDate = true;
    return false;
  }

  // The caller wants to handle missing module files.
  if (Missing && ReadResult == ASTReader::Missing) {
    *Missing = true;
    return false;
  }

  // The ASTReader didn't diagnose the error, so conservatively report it.
  if (ReadResult == ASTReader::Missing || !Diags.hasErrorOccurred())
    Diags.Report(ModuleNameLoc, diag::err_module_not_built)
      << Module->Name << SourceRange(ImportLoc, ModuleNameLoc);

  return false;
}

/// Compile a module in a separate compiler instance and read the AST,
/// returning true if the module compiles without errors.
static bool compileModuleAndReadASTImpl(CompilerInstance &ImportingInstance,
                                        SourceLocation ImportLoc,
                                        SourceLocation ModuleNameLoc,
                                        Module *Module,
                                        StringRef ModuleFileName) {
  auto Instance = ImportingInstance.cloneForModuleCompile(ModuleNameLoc, Module,
                                                          ModuleFileName);

  if (!ImportingInstance.compileModule(ModuleNameLoc,
                                       Module->getTopLevelModuleName(),
                                       ModuleFileName, *Instance)) {
    ImportingInstance.getDiagnostics().Report(ModuleNameLoc,
                                              diag::err_module_not_built)
        << Module->Name << SourceRange(ImportLoc, ModuleNameLoc);
    return false;
  }

  // The module is built successfully, we can update its timestamp now.
  if (ImportingInstance.getPreprocessor()
          .getHeaderSearchInfo()
          .getHeaderSearchOpts()
          .ModulesValidateOncePerBuildSession) {
    ImportingInstance.getModuleCache().updateModuleTimestamp(ModuleFileName);
  }

  return readASTAfterCompileModule(ImportingInstance, ImportLoc, ModuleNameLoc,
                                   Module, ModuleFileName,
                                   /*OutOfDate=*/nullptr, /*Missing=*/nullptr);
}

/// Compile a module in a separate compiler instance and read the AST,
/// returning true if the module compiles without errors, using a lock manager
/// to avoid building the same module in multiple compiler instances.
///
/// Uses a lock file manager and exponential backoff to reduce the chances that
/// multiple instances will compete to create the same module.  On timeout,
/// deletes the lock file in order to avoid deadlock from crashing processes or
/// bugs in the lock file manager.
static bool compileModuleAndReadASTBehindLock(
    CompilerInstance &ImportingInstance, SourceLocation ImportLoc,
    SourceLocation ModuleNameLoc, Module *Module, StringRef ModuleFileName) {
  DiagnosticsEngine &Diags = ImportingInstance.getDiagnostics();

  Diags.Report(ModuleNameLoc, diag::remark_module_lock)
      << ModuleFileName << Module->Name;

  auto &ModuleCache = ImportingInstance.getModuleCache();
  ModuleCache.prepareForGetLock(ModuleFileName);

  while (true) {
    auto Lock = ModuleCache.getLock(ModuleFileName);
    bool Owned;
    if (llvm::Error Err = Lock->tryLock().moveInto(Owned)) {
      // ModuleCache takes care of correctness and locks are only necessary for
      // performance. Fallback to building the module in case of any lock
      // related errors.
      Diags.Report(ModuleNameLoc, diag::remark_module_lock_failure)
          << Module->Name << toString(std::move(Err));
      return compileModuleAndReadASTImpl(ImportingInstance, ImportLoc,
                                         ModuleNameLoc, Module, ModuleFileName);
    }
    if (Owned) {
      // We're responsible for building the module ourselves.
      return compileModuleAndReadASTImpl(ImportingInstance, ImportLoc,
                                         ModuleNameLoc, Module, ModuleFileName);
    }

    // Someone else is responsible for building the module. Wait for them to
    // finish.
    switch (Lock->waitForUnlockFor(std::chrono::seconds(90))) {
    case llvm::WaitForUnlockResult::Success:
      break; // The interesting case.
    case llvm::WaitForUnlockResult::OwnerDied:
      continue; // try again to get the lock.
    case llvm::WaitForUnlockResult::Timeout:
      // Since the InMemoryModuleCache takes care of correctness, we try waiting
      // for someone else to complete the build so that it does not happen
      // twice. In case of timeout, build it ourselves.
      Diags.Report(ModuleNameLoc, diag::remark_module_lock_timeout)
          << Module->Name;
      // Clear the lock file so that future invocations can make progress.
      Lock->unsafeMaybeUnlock();
      continue;
    }

    // Read the module that was just written by someone else.
    bool OutOfDate = false;
    bool Missing = false;
    if (readASTAfterCompileModule(ImportingInstance, ImportLoc, ModuleNameLoc,
                                  Module, ModuleFileName, &OutOfDate, &Missing))
      return true;
    if (!OutOfDate && !Missing)
      return false;

    // The module may be missing or out of date in the presence of file system
    // races. It may also be out of date if one of its imports depends on header
    // search paths that are not consistent with this ImportingInstance.
    // Try again...
  }
}

/// Compile a module in a separate compiler instance and read the AST,
/// returning true if the module compiles without errors, potentially using a
/// lock manager to avoid building the same module in multiple compiler
/// instances.
static bool compileModuleAndReadAST(CompilerInstance &ImportingInstance,
                                    SourceLocation ImportLoc,
                                    SourceLocation ModuleNameLoc,
                                    Module *Module, StringRef ModuleFileName) {
  return ImportingInstance.getInvocation()
                 .getFrontendOpts()
                 .BuildingImplicitModuleUsesLock
             ? compileModuleAndReadASTBehindLock(ImportingInstance, ImportLoc,
                                                 ModuleNameLoc, Module,
                                                 ModuleFileName)
             : compileModuleAndReadASTImpl(ImportingInstance, ImportLoc,
                                           ModuleNameLoc, Module,
                                           ModuleFileName);
}

/// Diagnose differences between the current definition of the given
/// configuration macro and the definition provided on the command line.
static void checkConfigMacro(Preprocessor &PP, StringRef ConfigMacro,
                             Module *Mod, SourceLocation ImportLoc) {
  IdentifierInfo *Id = PP.getIdentifierInfo(ConfigMacro);
  SourceManager &SourceMgr = PP.getSourceManager();

  // If this identifier has never had a macro definition, then it could
  // not have changed.
  if (!Id->hadMacroDefinition())
    return;
  auto *LatestLocalMD = PP.getLocalMacroDirectiveHistory(Id);

  // Find the macro definition from the command line.
  MacroInfo *CmdLineDefinition = nullptr;
  for (auto *MD = LatestLocalMD; MD; MD = MD->getPrevious()) {
    // We only care about the predefines buffer.
    FileID FID = SourceMgr.getFileID(MD->getLocation());
    if (FID.isInvalid() || FID != PP.getPredefinesFileID())
      continue;
    if (auto *DMD = dyn_cast<DefMacroDirective>(MD))
      CmdLineDefinition = DMD->getMacroInfo();
    break;
  }

  auto *CurrentDefinition = PP.getMacroInfo(Id);
  if (CurrentDefinition == CmdLineDefinition) {
    // Macro matches. Nothing to do.
  } else if (!CurrentDefinition) {
    // This macro was defined on the command line, then #undef'd later.
    // Complain.
    PP.Diag(ImportLoc, diag::warn_module_config_macro_undef)
      << true << ConfigMacro << Mod->getFullModuleName();
    auto LatestDef = LatestLocalMD->getDefinition();
    assert(LatestDef.isUndefined() &&
           "predefined macro went away with no #undef?");
    PP.Diag(LatestDef.getUndefLocation(), diag::note_module_def_undef_here)
      << true;
    return;
  } else if (!CmdLineDefinition) {
    // There was no definition for this macro in the predefines buffer,
    // but there was a local definition. Complain.
    PP.Diag(ImportLoc, diag::warn_module_config_macro_undef)
      << false << ConfigMacro << Mod->getFullModuleName();
    PP.Diag(CurrentDefinition->getDefinitionLoc(),
            diag::note_module_def_undef_here)
      << false;
  } else if (!CurrentDefinition->isIdenticalTo(*CmdLineDefinition, PP,
                                               /*Syntactically=*/true)) {
    // The macro definitions differ.
    PP.Diag(ImportLoc, diag::warn_module_config_macro_undef)
      << false << ConfigMacro << Mod->getFullModuleName();
    PP.Diag(CurrentDefinition->getDefinitionLoc(),
            diag::note_module_def_undef_here)
      << false;
  }
}

static void checkConfigMacros(Preprocessor &PP, Module *M,
                              SourceLocation ImportLoc) {
  clang::Module *TopModule = M->getTopLevelModule();
  for (const StringRef ConMacro : TopModule->ConfigMacros) {
    checkConfigMacro(PP, ConMacro, M, ImportLoc);
  }
}

/// Write a new timestamp file with the given path.
static void writeTimestampFile(StringRef TimestampFile) {
  std::error_code EC;
  llvm::raw_fd_ostream Out(TimestampFile.str(), EC, llvm::sys::fs::OF_None);
}

/// Prune the module cache of modules that haven't been accessed in
/// a long time.
static void pruneModuleCache(const HeaderSearchOptions &HSOpts) {
  llvm::sys::fs::file_status StatBuf;
  llvm::SmallString<128> TimestampFile;
  TimestampFile = HSOpts.ModuleCachePath;
  assert(!TimestampFile.empty());
  llvm::sys::path::append(TimestampFile, "modules.timestamp");

  // Try to stat() the timestamp file.
  if (std::error_code EC = llvm::sys::fs::status(TimestampFile, StatBuf)) {
    // If the timestamp file wasn't there, create one now.
    if (EC == std::errc::no_such_file_or_directory) {
      writeTimestampFile(TimestampFile);
    }
    return;
  }

  // Check whether the time stamp is older than our pruning interval.
  // If not, do nothing.
  time_t TimeStampModTime =
      llvm::sys::toTimeT(StatBuf.getLastModificationTime());
  time_t CurrentTime = time(nullptr);
  if (CurrentTime - TimeStampModTime <= time_t(HSOpts.ModuleCachePruneInterval))
    return;

  // Write a new timestamp file so that nobody else attempts to prune.
  // There is a benign race condition here, if two Clang instances happen to
  // notice at the same time that the timestamp is out-of-date.
  writeTimestampFile(TimestampFile);

  // Walk the entire module cache, looking for unused module files and module
  // indices.
  std::error_code EC;
  for (llvm::sys::fs::directory_iterator Dir(HSOpts.ModuleCachePath, EC),
       DirEnd;
       Dir != DirEnd && !EC; Dir.increment(EC)) {
    // If we don't have a directory, there's nothing to look into.
    if (!llvm::sys::fs::is_directory(Dir->path()))
      continue;

    // Walk all of the files within this directory.
    for (llvm::sys::fs::directory_iterator File(Dir->path(), EC), FileEnd;
         File != FileEnd && !EC; File.increment(EC)) {
      // We only care about module and global module index files.
      StringRef Extension = llvm::sys::path::extension(File->path());
      if (Extension != ".pcm" && Extension != ".timestamp" &&
          llvm::sys::path::filename(File->path()) != "modules.idx")
        continue;

      // Look at this file. If we can't stat it, there's nothing interesting
      // there.
      if (llvm::sys::fs::status(File->path(), StatBuf))
        continue;

      // If the file has been used recently enough, leave it there.
      time_t FileAccessTime = llvm::sys::toTimeT(StatBuf.getLastAccessedTime());
      if (CurrentTime - FileAccessTime <=
              time_t(HSOpts.ModuleCachePruneAfter)) {
        continue;
      }

      // Remove the file.
      llvm::sys::fs::remove(File->path());

      // Remove the timestamp file.
      std::string TimpestampFilename = File->path() + ".timestamp";
      llvm::sys::fs::remove(TimpestampFilename);
    }

    // If we removed all of the files in the directory, remove the directory
    // itself.
    if (llvm::sys::fs::directory_iterator(Dir->path(), EC) ==
            llvm::sys::fs::directory_iterator() && !EC)
      llvm::sys::fs::remove(Dir->path());
  }
}

void CompilerInstance::createASTReader() {
  if (TheASTReader)
    return;

  if (!hasASTContext())
    createASTContext();

  // If we're implicitly building modules but not currently recursively
  // building a module, check whether we need to prune the module cache.
  if (getSourceManager().getModuleBuildStack().empty() &&
      !getPreprocessor().getHeaderSearchInfo().getModuleCachePath().empty() &&
      getHeaderSearchOpts().ModuleCachePruneInterval > 0 &&
      getHeaderSearchOpts().ModuleCachePruneAfter > 0) {
    pruneModuleCache(getHeaderSearchOpts());
  }

  HeaderSearchOptions &HSOpts = getHeaderSearchOpts();
  std::string Sysroot = HSOpts.Sysroot;
  const PreprocessorOptions &PPOpts = getPreprocessorOpts();
  const FrontendOptions &FEOpts = getFrontendOpts();
  std::unique_ptr<llvm::Timer> ReadTimer;

  if (timerGroup)
    ReadTimer = std::make_unique<llvm::Timer>("reading_modules",
                                              "Reading modules", *timerGroup);
  TheASTReader = llvm::makeIntrusiveRefCnt<ASTReader>(
      getPreprocessor(), getModuleCache(), &getASTContext(),
      getPCHContainerReader(), getCodeGenOpts(),
      getFrontendOpts().ModuleFileExtensions,
      Sysroot.empty() ? "" : Sysroot.c_str(),
      PPOpts.DisablePCHOrModuleValidation,
      /*AllowASTWithCompilerErrors=*/FEOpts.AllowPCMWithCompilerErrors,
      /*AllowConfigurationMismatch=*/false,
      +HSOpts.ModulesValidateSystemHeaders,
      +HSOpts.ModulesForceValidateUserHeaders,
      +HSOpts.ValidateASTInputFilesContent,
      +getFrontendOpts().UseGlobalModuleIndex, std::move(ReadTimer));
  if (hasASTConsumer()) {
    TheASTReader->setDeserializationListener(
        getASTConsumer().GetASTDeserializationListener());
    getASTContext().setASTMutationListener(
      getASTConsumer().GetASTMutationListener());
  }
  getASTContext().setExternalSource(TheASTReader);
  if (hasSema())
    TheASTReader->InitializeSema(getSema());
  if (hasASTConsumer())
    TheASTReader->StartTranslationUnit(&getASTConsumer());

  for (auto &Listener : DependencyCollectors)
    Listener->attachToASTReader(*TheASTReader);

  if (!FEOpts.ModuleLoadIgnoreCAS)
    TheASTReader->addListener(std::make_unique<CompileCacheASTReaderHelper>(
        getOrCreateObjectStore(), getOrCreateActionCache(), getModuleCache(),
        getDiagnostics()));
}

bool CompilerInstance::loadModuleFile(
    StringRef FileName, serialization::ModuleFile *&LoadedModuleFile) {
  llvm::Timer Timer;
  if (timerGroup)
    Timer.init("preloading." + FileName.str(), "Preloading " + FileName.str(),
               *timerGroup);
  llvm::TimeRegion TimeLoading(timerGroup ? &Timer : nullptr);

  // If we don't already have an ASTReader, create one now.
  if (!TheASTReader)
    createASTReader();

  // If -Wmodule-file-config-mismatch is mapped as an error or worse, allow the
  // ASTReader to diagnose it, since it can produce better errors that we can.
  bool ConfigMismatchIsRecoverable =
      getDiagnostics().getDiagnosticLevel(diag::warn_module_config_mismatch,
                                          SourceLocation())
        <= DiagnosticsEngine::Warning;

  auto Listener = std::make_unique<ReadModuleNames>(*PP);
  auto &ListenerRef = *Listener;
  ASTReader::ListenerScope ReadModuleNamesListener(*TheASTReader,
                                                   std::move(Listener));

  // Try to load the module file.
  switch (TheASTReader->ReadAST(
      FileName, serialization::MK_ExplicitModule, SourceLocation(),
      ConfigMismatchIsRecoverable ? ASTReader::ARR_ConfigurationMismatch : 0,
      &LoadedModuleFile)) {
  case ASTReader::Success:
    // We successfully loaded the module file; remember the set of provided
    // modules so that we don't try to load implicit modules for them.
    ListenerRef.registerAll();
    return true;

  case ASTReader::ConfigurationMismatch:
    // Ignore unusable module files.
    getDiagnostics().Report(SourceLocation(), diag::warn_module_config_mismatch)
        << FileName;
    // All modules provided by any files we tried and failed to load are now
    // unavailable; includes of those modules should now be handled textually.
    ListenerRef.markAllUnavailable();
    return true;

  default:
    return false;
  }
}

namespace {
enum ModuleSource {
  MS_ModuleNotFound,
  MS_ModuleCache,
  MS_PrebuiltModulePath,
  MS_ModuleBuildPragma
};
} // end namespace

/// Select a source for loading the named module and compute the filename to
/// load it from.
static ModuleSource selectModuleSource(
    Module *M, StringRef ModuleName, std::string &ModuleFilename,
    const std::map<std::string, std::string, std::less<>> &BuiltModules,
    HeaderSearch &HS) {
  assert(ModuleFilename.empty() && "Already has a module source?");

  // Check to see if the module has been built as part of this compilation
  // via a module build pragma.
  auto BuiltModuleIt = BuiltModules.find(ModuleName);
  if (BuiltModuleIt != BuiltModules.end()) {
    ModuleFilename = BuiltModuleIt->second;
    return MS_ModuleBuildPragma;
  }

  // Try to load the module from the prebuilt module path.
  const HeaderSearchOptions &HSOpts = HS.getHeaderSearchOpts();
  if (!HSOpts.PrebuiltModuleFiles.empty() ||
      !HSOpts.PrebuiltModulePaths.empty()) {
    ModuleFilename = HS.getPrebuiltModuleFileName(ModuleName);
    if (HSOpts.EnablePrebuiltImplicitModules && ModuleFilename.empty())
      ModuleFilename = HS.getPrebuiltImplicitModuleFileName(M);
    if (!ModuleFilename.empty())
      return MS_PrebuiltModulePath;
  }

  // Try to load the module from the module cache.
  if (M) {
    ModuleFilename = HS.getCachedModuleFileName(M);
    return MS_ModuleCache;
  }

  return MS_ModuleNotFound;
}

ModuleLoadResult CompilerInstance::findOrCompileModuleAndReadAST(
    StringRef ModuleName, SourceLocation ImportLoc,
    SourceLocation ModuleNameLoc, bool IsInclusionDirective) {
  // Search for a module with the given name.
  HeaderSearch &HS = PP->getHeaderSearchInfo();
  Module *M =
      HS.lookupModule(ModuleName, ImportLoc, true, !IsInclusionDirective);

  // Check for any configuration macros that have changed. This is done
  // immediately before potentially building a module in case this module
  // depends on having one of its configuration macros defined to successfully
  // build. If this is not done the user will never see the warning.
  if (M)
    checkConfigMacros(getPreprocessor(), M, ImportLoc);

  // Select the source and filename for loading the named module.
  std::string ModuleFilename;
  ModuleSource Source =
      selectModuleSource(M, ModuleName, ModuleFilename, BuiltModules, HS);
  if (Source == MS_ModuleNotFound) {
    // We can't find a module, error out here.
    getDiagnostics().Report(ModuleNameLoc, diag::err_module_not_found)
        << ModuleName << SourceRange(ImportLoc, ModuleNameLoc);
    return nullptr;
  }
  if (ModuleFilename.empty()) {
    if (M && M->HasIncompatibleModuleFile) {
      // We tried and failed to load a module file for this module. Fall
      // back to textual inclusion for its headers.
      return ModuleLoadResult::ConfigMismatch;
    }

    getDiagnostics().Report(ModuleNameLoc, diag::err_module_build_disabled)
        << ModuleName;
    return nullptr;
  }

  // Create an ASTReader on demand.
  if (!getASTReader())
    createASTReader();

  // Time how long it takes to load the module.
  llvm::Timer Timer;
  if (timerGroup)
    Timer.init("loading." + ModuleFilename, "Loading " + ModuleFilename,
               *timerGroup);
  llvm::TimeRegion TimeLoading(timerGroup ? &Timer : nullptr);
  llvm::TimeTraceScope TimeScope("Module Load", ModuleName);

  // Try to load the module file. If we are not trying to load from the
  // module cache, we don't know how to rebuild modules.
  unsigned ARRFlags = Source == MS_ModuleCache
                          ? ASTReader::ARR_OutOfDate | ASTReader::ARR_Missing |
                                ASTReader::ARR_TreatModuleWithErrorsAsOutOfDate
                          : Source == MS_PrebuiltModulePath
                                ? 0
                                : ASTReader::ARR_ConfigurationMismatch;
  switch (getASTReader()->ReadAST(ModuleFilename,
                                  Source == MS_PrebuiltModulePath
                                      ? serialization::MK_PrebuiltModule
                                      : Source == MS_ModuleBuildPragma
                                            ? serialization::MK_ExplicitModule
                                            : serialization::MK_ImplicitModule,
                                  ImportLoc, ARRFlags)) {
  case ASTReader::Success: {
    if (M)
      return M;
    assert(Source != MS_ModuleCache &&
           "missing module, but file loaded from cache");

    // A prebuilt module is indexed as a ModuleFile; the Module does not exist
    // until the first call to ReadAST.  Look it up now.
    M = HS.lookupModule(ModuleName, ImportLoc, true, !IsInclusionDirective);

    // Check whether M refers to the file in the prebuilt module path.
    if (M && M->getASTFile())
      if (auto ModuleFile = FileMgr->getOptionalFileRef(ModuleFilename)) {
        if (*ModuleFile == M->getASTFile())
          return M;
#if !defined(__APPLE__)
        // Workaround for ext4 file system. Also check bypass file if exists.
        if (auto Bypass = FileMgr->getBypassFile(*ModuleFile))
          if (*Bypass == M->getASTFile())
            return M;
#endif
      }

    getDiagnostics().Report(ModuleNameLoc, diag::err_module_prebuilt)
        << ModuleName;
    return ModuleLoadResult();
  }

  case ASTReader::OutOfDate:
  case ASTReader::Missing:
    // The most interesting case.
    break;

  case ASTReader::ConfigurationMismatch:
    if (Source == MS_PrebuiltModulePath)
      // FIXME: We shouldn't be setting HadFatalFailure below if we only
      // produce a warning here!
      getDiagnostics().Report(SourceLocation(),
                              diag::warn_module_config_mismatch)
          << ModuleFilename;
    // Fall through to error out.
    [[fallthrough]];
  case ASTReader::VersionMismatch:
  case ASTReader::HadErrors:
    ModuleLoader::HadFatalFailure = true;
    // FIXME: The ASTReader will already have complained, but can we shoehorn
    // that diagnostic information into a more useful form?
    return ModuleLoadResult();

  case ASTReader::Failure:
    ModuleLoader::HadFatalFailure = true;
    return ModuleLoadResult();
  }

  // ReadAST returned Missing or OutOfDate.
  if (Source != MS_ModuleCache) {
    // We don't know the desired configuration for this module and don't
    // necessarily even have a module map. Since ReadAST already produces
    // diagnostics for these two cases, we simply error out here.
    return ModuleLoadResult();
  }

  // The module file is missing or out-of-date. Build it.
  assert(M && "missing module, but trying to compile for cache");

  // Check whether there is a cycle in the module graph.
  ModuleBuildStack ModPath = getSourceManager().getModuleBuildStack();
  ModuleBuildStack::iterator Pos = ModPath.begin(), PosEnd = ModPath.end();
  for (; Pos != PosEnd; ++Pos) {
    if (Pos->first == ModuleName)
      break;
  }

  if (Pos != PosEnd) {
    SmallString<256> CyclePath;
    for (; Pos != PosEnd; ++Pos) {
      CyclePath += Pos->first;
      CyclePath += " -> ";
    }
    CyclePath += ModuleName;

    getDiagnostics().Report(ModuleNameLoc, diag::err_module_cycle)
        << ModuleName << CyclePath;
    return nullptr;
  }

  // Check whether we have already attempted to build this module (but failed).
  if (FailedModules.contains(ModuleName)) {
    getDiagnostics().Report(ModuleNameLoc, diag::err_module_not_built)
        << ModuleName << SourceRange(ImportLoc, ModuleNameLoc);
    return nullptr;
  }

  // Try to compile and then read the AST.
  if (!compileModuleAndReadAST(*this, ImportLoc, ModuleNameLoc, M,
                               ModuleFilename)) {
    assert(getDiagnostics().hasErrorOccurred() &&
           "undiagnosed error in compileModuleAndReadAST");
    FailedModules.insert(ModuleName);
    return nullptr;
  }

  // Okay, we've rebuilt and now loaded the module.
  return M;
}

ModuleLoadResult
CompilerInstance::loadModule(SourceLocation ImportLoc,
                             ModuleIdPath Path,
                             Module::NameVisibilityKind Visibility,
                             bool IsInclusionDirective) {
  // Determine what file we're searching from.
  StringRef ModuleName = Path[0].getIdentifierInfo()->getName();
  SourceLocation ModuleNameLoc = Path[0].getLoc();

  // If we've already handled this import, just return the cached result.
  // This one-element cache is important to eliminate redundant diagnostics
  // when both the preprocessor and parser see the same import declaration.
  if (ImportLoc.isValid() && LastModuleImportLoc == ImportLoc) {
    // Make the named module visible.
    if (LastModuleImportResult && ModuleName != getLangOpts().CurrentModule)
      TheASTReader->makeModuleVisible(LastModuleImportResult, Visibility,
                                      ImportLoc);
    return LastModuleImportResult;
  }

  // If we don't already have information on this module, load the module now.
  Module *Module = nullptr;
  ModuleMap &MM = getPreprocessor().getHeaderSearchInfo().getModuleMap();
  if (auto MaybeModule = MM.getCachedModuleLoad(*Path[0].getIdentifierInfo())) {
    // Use the cached result, which may be nullptr.
    Module = *MaybeModule;
    // Config macros are already checked before building a module, but they need
    // to be checked at each import location in case any of the config macros
    // have a new value at the current `ImportLoc`.
    if (Module)
      checkConfigMacros(getPreprocessor(), Module, ImportLoc);
  } else if (ModuleName == getLangOpts().CurrentModule) {
    // This is the module we're building.
    Module = PP->getHeaderSearchInfo().lookupModule(
        ModuleName, ImportLoc, /*AllowSearch*/ true,
        /*AllowExtraModuleMapSearch*/ !IsInclusionDirective);

    // Config macros do not need to be checked here for two reasons.
    // * This will always be textual inclusion, and thus the config macros
    //   actually do impact the content of the header.
    // * `Preprocessor::HandleHeaderIncludeOrImport` will never call this
    //   function as the `#include` or `#import` is textual.

    MM.cacheModuleLoad(*Path[0].getIdentifierInfo(), Module);
  } else {
    ModuleLoadResult Result = findOrCompileModuleAndReadAST(
        ModuleName, ImportLoc, ModuleNameLoc, IsInclusionDirective);
    if (!Result.isNormal())
      return Result;
    if (!Result)
      DisableGeneratingGlobalModuleIndex = true;
    Module = Result;
    MM.cacheModuleLoad(*Path[0].getIdentifierInfo(), Module);
  }

  // If we never found the module, fail.  Otherwise, verify the module and link
  // it up.
  if (!Module)
    return ModuleLoadResult();

  // Verify that the rest of the module path actually corresponds to
  // a submodule.
  bool MapPrivateSubModToTopLevel = false;
  for (unsigned I = 1, N = Path.size(); I != N; ++I) {
    StringRef Name = Path[I].getIdentifierInfo()->getName();
    clang::Module *Sub = Module->findSubmodule(Name);

    // If the user is requesting Foo.Private and it doesn't exist, try to
    // match Foo_Private and emit a warning asking for the user to write
    // @import Foo_Private instead. FIXME: remove this when existing clients
    // migrate off of Foo.Private syntax.
    if (!Sub && Name == "Private" && Module == Module->getTopLevelModule()) {
      SmallString<128> PrivateModule(Module->Name);
      PrivateModule.append("_Private");

      SmallVector<IdentifierLoc, 2> PrivPath;
      auto &II = PP->getIdentifierTable().get(
          PrivateModule, PP->getIdentifierInfo(Module->Name)->getTokenID());
      PrivPath.emplace_back(Path[0].getLoc(), &II);

      std::string FileName;
      // If there is a modulemap module or prebuilt module, load it.
      if (PP->getHeaderSearchInfo().lookupModule(PrivateModule, ImportLoc, true,
                                                 !IsInclusionDirective) ||
          selectModuleSource(nullptr, PrivateModule, FileName, BuiltModules,
                             PP->getHeaderSearchInfo()) != MS_ModuleNotFound)
        Sub = loadModule(ImportLoc, PrivPath, Visibility, IsInclusionDirective);
      if (Sub) {
        MapPrivateSubModToTopLevel = true;
        PP->markClangModuleAsAffecting(Module);
        if (!getDiagnostics().isIgnored(
                diag::warn_no_priv_submodule_use_toplevel, ImportLoc)) {
          getDiagnostics().Report(Path[I].getLoc(),
                                  diag::warn_no_priv_submodule_use_toplevel)
              << Path[I].getIdentifierInfo() << Module->getFullModuleName()
              << PrivateModule
              << SourceRange(Path[0].getLoc(), Path[I].getLoc())
              << FixItHint::CreateReplacement(SourceRange(Path[0].getLoc()),
                                              PrivateModule);
          getDiagnostics().Report(Sub->DefinitionLoc,
                                  diag::note_private_top_level_defined);
        }
      }
    }

    if (!Sub) {
      // Attempt to perform typo correction to find a module name that works.
      SmallVector<StringRef, 2> Best;
      unsigned BestEditDistance = (std::numeric_limits<unsigned>::max)();

      for (class Module *SubModule : Module->submodules()) {
        unsigned ED =
            Name.edit_distance(SubModule->Name,
                               /*AllowReplacements=*/true, BestEditDistance);
        if (ED <= BestEditDistance) {
          if (ED < BestEditDistance) {
            Best.clear();
            BestEditDistance = ED;
          }

          Best.push_back(SubModule->Name);
        }
      }

      // If there was a clear winner, user it.
      if (Best.size() == 1) {
        getDiagnostics().Report(Path[I].getLoc(),
                                diag::err_no_submodule_suggest)
            << Path[I].getIdentifierInfo() << Module->getFullModuleName()
            << Best[0] << SourceRange(Path[0].getLoc(), Path[I - 1].getLoc())
            << FixItHint::CreateReplacement(SourceRange(Path[I].getLoc()),
                                            Best[0]);

        Sub = Module->findSubmodule(Best[0]);
      }
    }

    if (!Sub) {
      // No submodule by this name. Complain, and don't look for further
      // submodules.
      getDiagnostics().Report(Path[I].getLoc(), diag::err_no_submodule)
          << Path[I].getIdentifierInfo() << Module->getFullModuleName()
          << SourceRange(Path[0].getLoc(), Path[I - 1].getLoc());
      break;
    }

    Module = Sub;
  }

  // Make the named module visible, if it's not already part of the module
  // we are parsing.
  if (ModuleName != getLangOpts().CurrentModule) {
    if (!Module->IsFromModuleFile && !MapPrivateSubModToTopLevel) {
      // We have an umbrella header or directory that doesn't actually include
      // all of the headers within the directory it covers. Complain about
      // this missing submodule and recover by forgetting that we ever saw
      // this submodule.
      // FIXME: Should we detect this at module load time? It seems fairly
      // expensive (and rare).
      getDiagnostics().Report(ImportLoc, diag::warn_missing_submodule)
          << Module->getFullModuleName()
          << SourceRange(Path.front().getLoc(), Path.back().getLoc());

      Module->IsInferredMissingFromUmbrellaHeader = true;

      return ModuleLoadResult(Module, ModuleLoadResult::MissingExpected);
    }

    // Check whether this module is available.
    if (Preprocessor::checkModuleIsAvailable(getLangOpts(), getTarget(),
                                             *Module, getDiagnostics())) {
      getDiagnostics().Report(ImportLoc, diag::note_module_import_here)
          << SourceRange(Path.front().getLoc(), Path.back().getLoc());
      LastModuleImportLoc = ImportLoc;
      LastModuleImportResult = ModuleLoadResult();
      return ModuleLoadResult();
    }

    TheASTReader->makeModuleVisible(Module, Visibility, ImportLoc);
  }

  // Resolve any remaining module using export_as for this one.
  getPreprocessor()
      .getHeaderSearchInfo()
      .getModuleMap()
      .resolveLinkAsDependencies(Module->getTopLevelModule());

  LastModuleImportLoc = ImportLoc;
  LastModuleImportResult = ModuleLoadResult(Module);
  return LastModuleImportResult;
}

void CompilerInstance::createModuleFromSource(SourceLocation ImportLoc,
                                              StringRef ModuleName,
                                              StringRef Source) {
  // Avoid creating filenames with special characters.
  SmallString<128> CleanModuleName(ModuleName);
  for (auto &C : CleanModuleName)
    if (!isAlphanumeric(C))
      C = '_';

  // FIXME: Using a randomized filename here means that our intermediate .pcm
  // output is nondeterministic (as .pcm files refer to each other by name).
  // Can this affect the output in any way?
  SmallString<128> ModuleFileName;
  if (std::error_code EC = llvm::sys::fs::createTemporaryFile(
          CleanModuleName, "pcm", ModuleFileName)) {
    getDiagnostics().Report(ImportLoc, diag::err_fe_unable_to_open_output)
        << ModuleFileName << EC.message();
    return;
  }
  std::string ModuleMapFileName = (CleanModuleName + ".map").str();

  FrontendInputFile Input(
      ModuleMapFileName,
      InputKind(getLanguageFromOptions(Invocation->getLangOpts()),
                InputKind::ModuleMap, /*Preprocessed*/true));

  std::string NullTerminatedSource(Source.str());

  auto Other = cloneForModuleCompileImpl(ImportLoc, ModuleName, Input,
                                         StringRef(), ModuleFileName);

  // Create a virtual file containing our desired source.
  // FIXME: We shouldn't need to do this.
  FileEntryRef ModuleMapFile = Other->getFileManager().getVirtualFileRef(
      ModuleMapFileName, NullTerminatedSource.size(), 0);
  Other->getSourceManager().overrideFileContents(
      ModuleMapFile, llvm::MemoryBuffer::getMemBuffer(NullTerminatedSource));

  Other->BuiltModules = std::move(BuiltModules);
  Other->DeleteBuiltModules = false;

  // Build the module, inheriting any modules that we've built locally.
  bool Success = compileModule(ImportLoc, ModuleName, ModuleFileName, *Other);

  BuiltModules = std::move(Other->BuiltModules);

  if (Success) {
    BuiltModules[std::string(ModuleName)] = std::string(ModuleFileName);
    llvm::sys::RemoveFileOnSignal(ModuleFileName);
  }
}

void CompilerInstance::makeModuleVisible(Module *Mod,
                                         Module::NameVisibilityKind Visibility,
                                         SourceLocation ImportLoc) {
  if (!TheASTReader)
    createASTReader();
  if (!TheASTReader)
    return;

  TheASTReader->makeModuleVisible(Mod, Visibility, ImportLoc);
}

GlobalModuleIndex *CompilerInstance::loadGlobalModuleIndex(
    SourceLocation TriggerLoc) {
  if (getPreprocessor().getHeaderSearchInfo().getModuleCachePath().empty())
    return nullptr;
  if (!TheASTReader)
    createASTReader();
  // Can't do anything if we don't have the module manager.
  if (!TheASTReader)
    return nullptr;
  // Get an existing global index.  This loads it if not already
  // loaded.
  TheASTReader->loadGlobalIndex();
  GlobalModuleIndex *GlobalIndex = TheASTReader->getGlobalIndex();
  // If the global index doesn't exist, create it.
  if (!GlobalIndex && shouldBuildGlobalModuleIndex() && hasFileManager() &&
      hasPreprocessor()) {
    llvm::sys::fs::create_directories(
      getPreprocessor().getHeaderSearchInfo().getModuleCachePath());
    if (llvm::Error Err = GlobalModuleIndex::writeIndex(
            getFileManager(), getPCHContainerReader(),
            getPreprocessor().getHeaderSearchInfo().getModuleCachePath())) {
      // FIXME this drops the error on the floor. This code is only used for
      // typo correction and drops more than just this one source of errors
      // (such as the directory creation failure above). It should handle the
      // error.
      consumeError(std::move(Err));
      return nullptr;
    }
    TheASTReader->resetForReload();
    TheASTReader->loadGlobalIndex();
    GlobalIndex = TheASTReader->getGlobalIndex();
  }
  // For finding modules needing to be imported for fixit messages,
  // we need to make the global index cover all modules, so we do that here.
  if (!HaveFullGlobalModuleIndex && GlobalIndex && !buildingModule()) {
    ModuleMap &MMap = getPreprocessor().getHeaderSearchInfo().getModuleMap();
    bool RecreateIndex = false;
    for (ModuleMap::module_iterator I = MMap.module_begin(),
        E = MMap.module_end(); I != E; ++I) {
      Module *TheModule = I->second;
      OptionalFileEntryRef Entry = TheModule->getASTFile();
      if (!Entry) {
        SmallVector<IdentifierLoc, 2> Path;
        Path.emplace_back(TriggerLoc,
                          getPreprocessor().getIdentifierInfo(TheModule->Name));
        std::reverse(Path.begin(), Path.end());
        // Load a module as hidden.  This also adds it to the global index.
        loadModule(TheModule->DefinitionLoc, Path, Module::Hidden, false);
        RecreateIndex = true;
      }
    }
    if (RecreateIndex) {
      if (llvm::Error Err = GlobalModuleIndex::writeIndex(
              getFileManager(), getPCHContainerReader(),
              getPreprocessor().getHeaderSearchInfo().getModuleCachePath())) {
        // FIXME As above, this drops the error on the floor.
        consumeError(std::move(Err));
        return nullptr;
      }
      TheASTReader->resetForReload();
      TheASTReader->loadGlobalIndex();
      GlobalIndex = TheASTReader->getGlobalIndex();
    }
    HaveFullGlobalModuleIndex = true;
  }
  return GlobalIndex;
}

// Check global module index for missing imports.
bool
CompilerInstance::lookupMissingImports(StringRef Name,
                                       SourceLocation TriggerLoc) {
  // Look for the symbol in non-imported modules, but only if an error
  // actually occurred.
  if (!buildingModule()) {
    // Load global module index, or retrieve a previously loaded one.
    GlobalModuleIndex *GlobalIndex = loadGlobalModuleIndex(
      TriggerLoc);

    // Only if we have a global index.
    if (GlobalIndex) {
      GlobalModuleIndex::HitSet FoundModules;

      // Find the modules that reference the identifier.
      // Note that this only finds top-level modules.
      // We'll let diagnoseTypo find the actual declaration module.
      if (GlobalIndex->lookupIdentifier(Name, FoundModules))
        return true;
    }
  }

  return false;
}
void CompilerInstance::resetAndLeakSema() { llvm::BuryPointer(takeSema()); }

void CompilerInstance::setExternalSemaSource(
    IntrusiveRefCntPtr<ExternalSemaSource> ESS) {
  ExternalSemaSrc = std::move(ESS);
}

static bool addCachedModuleFileToInMemoryCache(
    StringRef Path, StringRef CacheKey, StringRef Provider,
    cas::ObjectStore &CAS, cas::ActionCache &Cache,
    ModuleCache &ModCache, DiagnosticsEngine &Diags) {

  if (ModCache.getInMemoryModuleCache().lookupPCM(Path))
    return false;

  auto ID = CAS.parseID(CacheKey);
  if (!ID) {
    Diags.Report(diag::err_cas_unloadable_module)
        << Path << CacheKey << ID.takeError();
    return true;
  }

  auto Value = Cache.get(*ID);
  if (!Value) {
    Diags.Report(diag::err_cas_unloadable_module)
        << Path << CacheKey << Value.takeError();
    return true;
  }
  if (!*Value) {
    auto Diag = Diags.Report(diag::err_cas_missing_module)
                << Path << CacheKey;
    std::string ErrStr("expected to be produced by:\n");
    llvm::raw_string_ostream Err(ErrStr);
    if (auto E = printCompileJobCacheKey(CAS, *ID, Err)) {
      // Ignore the error and skip printing the cache key. The cache key can
      // be setup by a different compiler that is using an unknown schema.
      llvm::consumeError(std::move(E));
      Diag << "module file is not available in the CAS";
    } else
      Diag << Err.str();

    return true;
  }
  auto ValueRef = CAS.getReference(**Value);
  if (!ValueRef) {
    Diags.Report(diag::err_cas_unloadable_module)
        << Path << CacheKey << "result module cannot be loaded from CAS";

    return true;
  }

  std::optional<cas::CompileJobCacheResult> Result;
  cas::CompileJobResultSchema Schema(CAS);
  if (llvm::Error E = Schema.load(*ValueRef).moveInto(Result)) {
    Diags.Report(diag::err_cas_unloadable_module)
        << Path << CacheKey << std::move(E);
    return true;
  }
  auto Output =
      Result->getOutput(cas::CompileJobCacheResult::OutputKind::MainOutput);
  if (!Output)
    llvm::report_fatal_error("missing main output");
  // FIXME: We wait to materialize each module file before proceeding, which
  // introduces latency for a network CAS. Instead we should collect all the
  // module keys and materialize them concurrently using \c getProxyFuture, for
  // better network utilization.
  auto OutputProxy = CAS.getProxy(Output->Object);
  if (!OutputProxy) {
    Diags.Report(diag::err_cas_unloadable_module)
        << Path << CacheKey << OutputProxy.takeError();
    return true;
  }

  ModCache.getInMemoryModuleCache().addPCM(Path,
                                           OutputProxy->getMemoryBuffer());
  return false;
}

bool CompilerInstance::addCachedModuleFile(StringRef Path, StringRef CacheKey,
                                           StringRef Provider) {
  return addCachedModuleFileToInMemoryCache(
      Path, CacheKey, Provider, getOrCreateObjectStore(),
      getOrCreateActionCache(), getModuleCache(), getDiagnostics());
}

bool CompileCacheASTReaderHelper::readModuleCacheKey(StringRef ModuleName,
                                                     StringRef Filename,
                                                     StringRef CacheKey) {
  // FIXME: add name/path of the importing module?
  return addCachedModuleFileToInMemoryCache(
      Filename, CacheKey, "imported module", CAS, Cache, ModCache, Diags);
}

/// Verify that ID is in the CAS. Otherwise the module cache probably was
/// created with a different CAS.
bool CompileCacheASTReaderHelper::checkCASID(bool Complain, StringRef RootID,
                                             unsigned ParseDiagID,
                                             unsigned MissingDiagID) {
  std::optional<cas::CASID> ID;
  if (errorToBool(CAS.parseID(RootID).moveInto(ID))) {
    if (Complain)
      Diags.Report(ParseDiagID) << RootID;
    return true;
  }
  if (errorToBool(CAS.getProxy(*ID).takeError())) {
    if (Complain) {
      Diags.Report(MissingDiagID) << RootID;
    }
    return true;
  }
  return false;
}

bool CompileCacheASTReaderHelper::readCASFileSystemRootID(StringRef RootID,
                                                          bool Complain) {
  return checkCASID(Complain, RootID, diag::err_cas_cannot_parse_root_id,
                    diag::err_cas_missing_root_id);
}

bool CompileCacheASTReaderHelper::readIncludeTreeID(StringRef ID,
                                                    bool Complain) {
  // Verify that ID is in the CAS. Otherwise the module cache probably was
  // created with a different CAS.
  return checkCASID(Complain, ID, diag::err_cas_cannot_parse_include_tree_id,
                    diag::err_cas_missing_include_tree_id);
}
