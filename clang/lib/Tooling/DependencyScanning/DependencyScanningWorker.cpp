//===- DependencyScanningWorker.cpp - clang-scan-deps worker --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/DependencyScanning/DependencyScanningWorker.h"
#include "clang/Basic/DiagnosticCAS.h"
#include "clang/Basic/DiagnosticFrontend.h"
#include "clang/CodeGen/ObjectFilePCHContainerOperations.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Job.h"
#include "clang/Driver/Tool.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/MultiplexConsumer.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Frontend/Utils.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "clang/Tooling/DependencyScanning/DependencyScanningService.h"
#include "clang/Tooling/DependencyScanning/ModuleDepCollector.h"
#include "clang/Tooling/DependencyScanning/ScanAndUpdateArgs.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/CAS/CachingOnDiskFileSystem.h"
#include "llvm/CAS/ObjectStore.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/PrefixMapper.h"

using namespace clang;
using namespace tooling;
using namespace dependencies;
using llvm::Error;

namespace {

/// Forwards the gatherered dependencies to the consumer.
class DependencyConsumerForwarder : public DependencyFileGenerator {
public:
  DependencyConsumerForwarder(std::unique_ptr<DependencyOutputOptions> Opts,
                              StringRef WorkingDirectory, DependencyConsumer &C,
                              bool EmitDependencyFile)
      : DependencyFileGenerator(*Opts), WorkingDirectory(WorkingDirectory),
        Opts(std::move(Opts)), C(C), EmitDependencyFile(EmitDependencyFile) {}

  void finishedMainFile(DiagnosticsEngine &Diags) override {
    C.handleDependencyOutputOpts(*Opts);
    llvm::SmallString<256> CanonPath;
    for (const auto &File : getDependencies()) {
      CanonPath = File;
      llvm::sys::path::remove_dots(CanonPath, /*remove_dot_dot=*/true);
      llvm::sys::fs::make_absolute(WorkingDirectory, CanonPath);
      C.handleFileDependency(CanonPath);
    }
    if (EmitDependencyFile)
      DependencyFileGenerator::finishedMainFile(Diags);
  }

private:
  StringRef WorkingDirectory;
  std::unique_ptr<DependencyOutputOptions> Opts;
  DependencyConsumer &C;
  bool EmitDependencyFile = false;
};

using PrebuiltModuleFilesT = decltype(HeaderSearchOptions::PrebuiltModuleFiles);

/// A listener that collects the imported modules and optionally the input
/// files.
class PrebuiltModuleListener : public ASTReaderListener {
public:
  PrebuiltModuleListener(CompilerInstance &CI,
                         PrebuiltModuleFilesT &PrebuiltModuleFiles,
                         llvm::StringSet<> &InputFiles, bool VisitInputFiles,
                         llvm::SmallVector<std::string> &NewModuleFiles)
      : CI(CI), PrebuiltModuleFiles(PrebuiltModuleFiles),
        InputFiles(InputFiles), VisitInputFiles(VisitInputFiles),
        NewModuleFiles(NewModuleFiles) {}

  bool needsImportVisitation() const override { return true; }
  bool needsInputFileVisitation() override { return VisitInputFiles; }
  bool needsSystemInputFileVisitation() override { return VisitInputFiles; }

  void visitImport(StringRef ModuleName, StringRef Filename) override {
    if (PrebuiltModuleFiles.insert({ModuleName.str(), Filename.str()}).second)
      NewModuleFiles.push_back(Filename.str());
  }

  bool visitInputFile(StringRef Filename, bool isSystem, bool isOverridden,
                      bool isExplicitModule) override {
    InputFiles.insert(Filename);
    return true;
  }

  bool readModuleCacheKey(StringRef ModuleName, StringRef Filename,
                          StringRef CacheKey) override {
    CI.getFrontendOpts().ModuleCacheKeys.emplace_back(std::string(Filename),
                                                      std::string(CacheKey));
    // FIXME: add name/path of the importing module?
    return CI.addCachedModuleFile(Filename, CacheKey, "imported module");
  }

private:
  CompilerInstance &CI;
  PrebuiltModuleFilesT &PrebuiltModuleFiles;
  llvm::StringSet<> &InputFiles;
  bool VisitInputFiles;
  llvm::SmallVector<std::string> &NewModuleFiles;
};

/// Visit the given prebuilt module and collect all of the modules it
/// transitively imports and contributing input files.
static void visitPrebuiltModule(StringRef PrebuiltModuleFilename,
                                CompilerInstance &CI,
                                PrebuiltModuleFilesT &ModuleFiles,
                                llvm::StringSet<> &InputFiles,
                                bool VisitInputFiles) {
  // List of module files to be processed.
  llvm::SmallVector<std::string> Worklist{PrebuiltModuleFilename.str()};
  PrebuiltModuleListener Listener(CI, ModuleFiles, InputFiles, VisitInputFiles,
                                  Worklist);

  while (!Worklist.empty())
    ASTReader::readASTFileControlBlock(
        Worklist.pop_back_val(), CI.getFileManager(), CI.getModuleCache(),
        CI.getPCHContainerReader(),
        /*FindModuleFileExtensions=*/false, Listener,
        /*ValidateDiagnosticOptions=*/false);
}

/// Transform arbitrary file name into an object-like file name.
static std::string makeObjFileName(StringRef FileName) {
  SmallString<128> ObjFileName(FileName);
  llvm::sys::path::replace_extension(ObjFileName, "o");
  return std::string(ObjFileName.str());
}

/// Deduce the dependency target based on the output file and input files.
static std::string
deduceDepTarget(const std::string &OutputFile,
                const SmallVectorImpl<FrontendInputFile> &InputFiles) {
  if (OutputFile != "-")
    return OutputFile;

  if (InputFiles.empty() || !InputFiles.front().isFile())
    return "clang-scan-deps\\ dependency";

  return makeObjFileName(InputFiles.front().getFile());
}

/// Sanitize diagnostic options for dependency scan.
static void sanitizeDiagOpts(DiagnosticOptions &DiagOpts) {
  // Don't print 'X warnings and Y errors generated'.
  DiagOpts.ShowCarets = false;
  // Don't write out diagnostic file.
  DiagOpts.DiagnosticSerializationFile.clear();
  // Don't emit warnings as errors (and all other warnings too).
  DiagOpts.IgnoreWarnings = true;
}

struct IncludeTreePPCallbacks : public PPCallbacks {
  PPIncludeActionsConsumer &Consumer;
  Preprocessor &PP;

public:
  IncludeTreePPCallbacks(PPIncludeActionsConsumer &Consumer, Preprocessor &PP)
      : Consumer(Consumer), PP(PP) {}

  void LexedFileChanged(FileID FID, LexedFileChangeReason Reason,
                        SrcMgr::CharacteristicKind FileType, FileID PrevFID,
                        SourceLocation Loc) override {
    switch (Reason) {
    case LexedFileChangeReason::EnterFile:
      Consumer.enteredInclude(PP, FID);
      break;
    case LexedFileChangeReason::ExitFile: {
      Consumer.exitedInclude(PP, FID, PrevFID, Loc);
      break;
    }
    }
  }

  void HasInclude(SourceLocation Loc, StringRef FileName, bool IsAngled,
                  Optional<FileEntryRef> File,
                  SrcMgr::CharacteristicKind FileType) override {
    Consumer.handleHasIncludeCheck(PP, File.has_value());
  }
};

class IncludeTreeCollector : public DependencyFileGenerator {
  PPIncludeActionsConsumer &Consumer;
  std::unique_ptr<DependencyOutputOptions> Opts;
  bool EmitDependencyFile = false;
  llvm::PrefixMapper ReverseMapper;

public:
  IncludeTreeCollector(PPIncludeActionsConsumer &Consumer,
                       std::unique_ptr<DependencyOutputOptions> Opts,
                       bool EmitDependencyFile)
      : DependencyFileGenerator(*Opts), Consumer(Consumer),
        Opts(std::move(Opts)), EmitDependencyFile(EmitDependencyFile) {}

  Error initialize(const CompilerInvocation &CI,
                   const DepscanPrefixMapping &PrefixMapping) {
    llvm::PrefixMapper Mapper;
    if (Error E = PrefixMapping.configurePrefixMapper(CI, Mapper))
      return E;
    if (Mapper.empty())
      return Error::success();

    ReverseMapper.addInverseRange(Mapper.getMappings());
    ReverseMapper.sort();
    return Error::success();
  }

  void attachToPreprocessor(Preprocessor &PP) override {
    PP.addPPCallbacks(std::make_unique<IncludeTreePPCallbacks>(Consumer, PP));
    DependencyFileGenerator::attachToPreprocessor(PP);
  }

  void maybeAddDependency(StringRef Filename, bool FromModule, bool IsSystem,
                          bool IsModuleFile, bool IsMissing) override {
    if (ReverseMapper.empty())
      return DependencyFileGenerator::maybeAddDependency(
          Filename, FromModule, IsSystem, IsModuleFile, IsMissing);

    // We may get canonicalized paths if prefix headers/PCH are used, so make
    // sure to remap them back to original source paths.
    SmallString<256> New{Filename};
    ReverseMapper.mapInPlace(New);
    return DependencyFileGenerator::maybeAddDependency(
        New, FromModule, IsSystem, IsModuleFile, IsMissing);
  }

  void finishedMainFile(DiagnosticsEngine &Diags) override {
    if (EmitDependencyFile)
      DependencyFileGenerator::finishedMainFile(Diags);
  }
};

/// See \c WrapScanModuleBuildAction.
class WrapScanModuleBuildConsumer : public ASTConsumer {
public:
  WrapScanModuleBuildConsumer(CompilerInstance &CI, Module &M,
                              llvm::cas::CachingOnDiskFileSystem &FS)
      : M(M), FS(FS) {}

  void HandleTranslationUnit(ASTContext &Ctx) override {
    auto Tree = FS.createTreeFromNewAccesses();
    if (Tree)
      M.setCASFileSystemRootID(Tree->getID().toString());
    else
      Ctx.getDiagnostics().Report(diag::err_cas_depscan_failed)
          << Tree.takeError();
  }

private:
  Module &M;
  llvm::cas::CachingOnDiskFileSystem &FS;
};

/// A wrapper for implicit module build actions in the scanner.
///
/// Captures per-module information such as the CAS filesystem root ID for
/// module inputs.
class WrapScanModuleBuildAction : public WrapperFrontendAction {
public:
  WrapScanModuleBuildAction(
      std::unique_ptr<FrontendAction> WrappedAction,
      IntrusiveRefCntPtr<llvm::cas::CachingOnDiskFileSystem> FS)
      : WrapperFrontendAction(std::move(WrappedAction)), FS(std::move(FS)) {
    assert(this->FS);
  }

private:
  bool BeginInvocation(CompilerInstance &CI) override {
    // FIXME: exclude unnecessary files such as implicit module pcms from the
    // scanner itself. These are wasteful and cause spurious cache misses.
    FS->trackNewAccesses();
    // Normally this would be looked up while creating the VFS, but implicit
    // modules share their VFS.
    for (const auto &File : CI.getHeaderSearchOpts().VFSOverlayFiles)
      (void)FS->status(File);
    // Exclude the module cache from tracking. The implicit build pcms should
    // not be needed after scanning.
    if (!CI.getHeaderSearchOpts().ModuleCachePath.empty())
      (void)FS->excludeFromTracking(CI.getHeaderSearchOpts().ModuleCachePath);
    return WrapperFrontendAction::BeginInvocation(CI);
  }

  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef InFile) override {
    auto OtherConsumer = WrapperFrontendAction::CreateASTConsumer(CI, InFile);
    if (!OtherConsumer)
      return nullptr;
    Module *M = CI.getPreprocessor().getCurrentModule();
    assert(M && "WrapScanModuleBuildAction should only be used with module");
    if (!M)
      return OtherConsumer;
    auto Consumer = std::make_unique<WrapScanModuleBuildConsumer>(CI, *M, *FS);
    std::vector<std::unique_ptr<ASTConsumer>> Consumers;
    Consumers.push_back(std::move(Consumer));
    Consumers.push_back(std::move(OtherConsumer));
    return std::make_unique<MultiplexConsumer>(std::move(Consumers));
  }

private:
  llvm::IntrusiveRefCntPtr<llvm::cas::CachingOnDiskFileSystem> FS;
};

/// A clang tool that runs the preprocessor in a mode that's optimized for
/// dependency scanning for the given compiler invocation.
class DependencyScanningAction : public tooling::ToolAction {
public:
  DependencyScanningAction(
      StringRef WorkingDirectory, DependencyConsumer &Consumer,
      llvm::IntrusiveRefCntPtr<DependencyScanningWorkerFilesystem> DepFS,
      llvm::IntrusiveRefCntPtr<DependencyScanningCASFilesystem> DepCASFS,
      llvm::IntrusiveRefCntPtr<llvm::cas::CachingOnDiskFileSystem> CacheFS,
      ScanningOutputFormat Format, bool OptimizeArgs, bool EagerLoadModules,
      bool DisableFree, bool EmitDependencyFile,
      bool DiagGenerationAsCompilation, const CASOptions &CASOpts,
      RemapPathCallback RemapPath,
      llvm::Optional<StringRef> ModuleName = None,
      raw_ostream *VerboseOS = nullptr)
      : WorkingDirectory(WorkingDirectory), Consumer(Consumer),
        DepFS(std::move(DepFS)), DepCASFS(std::move(DepCASFS)),
        CacheFS(std::move(CacheFS)), Format(Format), OptimizeArgs(OptimizeArgs),
        EagerLoadModules(EagerLoadModules), DisableFree(DisableFree),
        CASOpts(CASOpts), RemapPath(RemapPath),
        EmitDependencyFile(EmitDependencyFile),
        DiagGenerationAsCompilation(DiagGenerationAsCompilation),
        ModuleName(ModuleName), VerboseOS(VerboseOS) {}

  bool runInvocation(std::shared_ptr<CompilerInvocation> Invocation,
                     FileManager *FileMgr,
                     std::shared_ptr<PCHContainerOperations> PCHContainerOps,
                     DiagnosticConsumer *DiagConsumer) override {
    // Make a deep copy of the original Clang invocation.
    CompilerInvocation OriginalInvocation(*Invocation);
    // Restore the value of DisableFree, which may be modified by Tooling.
    OriginalInvocation.getFrontendOpts().DisableFree = DisableFree;

    if (Scanned) {
      // Scanning runs once for the first -cc1 invocation in a chain of driver
      // jobs. For any dependent jobs, reuse the scanning result and just
      // update the LastCC1Arguments to correspond to the new invocation.
      // FIXME: to support multi-arch builds, each arch requires a separate scan
      setLastCC1Arguments(std::move(OriginalInvocation));
      return true;
    }

    Scanned = true;

    // Create a compiler instance to handle the actual work.
    ScanInstanceStorage.emplace(std::move(PCHContainerOps));
    CompilerInstance &ScanInstance = *ScanInstanceStorage;
    ScanInstance.setInvocation(std::move(Invocation));
    ScanInstance.getInvocation().getCASOpts() = CASOpts;

    if (CacheFS) {
      CacheFS->trackNewAccesses();
      CacheFS->setCurrentWorkingDirectory(WorkingDirectory);
      // Enable caching in the resulting commands.
      ScanInstance.getFrontendOpts().CacheCompileJob = true;
    }

    // Create the compiler's actual diagnostics engine.
    if (!DiagGenerationAsCompilation)
      sanitizeDiagOpts(ScanInstance.getDiagnosticOpts());
    ScanInstance.createDiagnostics(DiagConsumer, /*ShouldOwnClient=*/false);
    if (!ScanInstance.hasDiagnostics())
      return false;
    if (VerboseOS)
      ScanInstance.setVerboseOutputStream(*VerboseOS);

    ScanInstance.getPreprocessorOpts().AllowPCHWithDifferentModulesCachePath =
        true;

    ScanInstance.getFrontendOpts().GenerateGlobalModuleIndex = false;
    ScanInstance.getFrontendOpts().UseGlobalModuleIndex = false;
    ScanInstance.getFrontendOpts().ModulesShareFileManager = false;

    ScanInstance.setFileManager(FileMgr);
    // Support for virtual file system overlays.
    FileMgr->setVirtualFileSystem(createVFSFromCompilerInvocation(
        ScanInstance.getInvocation(), ScanInstance.getDiagnostics(),
        FileMgr->getVirtualFileSystemPtr()));

    ScanInstance.createSourceManager(*FileMgr);

    llvm::StringSet<> PrebuiltModulesInputFiles;
    // Store the list of prebuilt module files into header search options. This
    // will prevent the implicit build to create duplicate modules and will
    // force reuse of the existing prebuilt module files instead.
    if (!ScanInstance.getPreprocessorOpts().ImplicitPCHInclude.empty())
      visitPrebuiltModule(
          ScanInstance.getPreprocessorOpts().ImplicitPCHInclude, ScanInstance,
          ScanInstance.getHeaderSearchOpts().PrebuiltModuleFiles,
          PrebuiltModulesInputFiles,
          /*VisitInputFiles=*/getDepScanFS() != nullptr);

    // Use the dependency scanning optimized file system if requested to do so.
    if (DepFS) {
      llvm::IntrusiveRefCntPtr<DependencyScanningWorkerFilesystem> LocalDepFS =
          DepFS;
      ScanInstance.getPreprocessorOpts().DependencyDirectivesForFile =
          [LocalDepFS = std::move(LocalDepFS)](FileEntryRef File)
          -> Optional<ArrayRef<dependency_directives_scan::Directive>> {
        if (llvm::ErrorOr<EntryRef> Entry =
                LocalDepFS->getOrCreateFileSystemEntry(File.getName()))
          return Entry->getDirectiveTokens();
        return None;
      };
    }
    // CAS Implementation.
    if (DepCASFS) {
      // Support for virtual file system overlays on top of the caching
      // filesystem.
      FileMgr->setVirtualFileSystem(createVFSFromCompilerInvocation(
          ScanInstance.getInvocation(), ScanInstance.getDiagnostics(),
          DepCASFS));

      llvm::IntrusiveRefCntPtr<DependencyScanningCASFilesystem> LocalDepCASFS =
          DepCASFS;
      ScanInstance.getPreprocessorOpts().DependencyDirectivesForFile =
          [LocalDepCASFS = std::move(LocalDepCASFS)](FileEntryRef File)
          -> Optional<ArrayRef<dependency_directives_scan::Directive>> {
        return LocalDepCASFS->getDirectiveTokens(File.getName());
      };
    }

    // Create the dependency collector that will collect the produced
    // dependencies.
    //
    // This also moves the existing dependency output options from the
    // invocation to the collector. The options in the invocation are reset,
    // which ensures that the compiler won't create new dependency collectors,
    // and thus won't write out the extra '.d' files to disk.
    auto Opts = std::make_unique<DependencyOutputOptions>();
    std::swap(*Opts, ScanInstance.getInvocation().getDependencyOutputOpts());
    // We need at least one -MT equivalent for the generator of make dependency
    // files to work.
    if (Opts->Targets.empty())
      Opts->Targets = {
          deduceDepTarget(ScanInstance.getFrontendOpts().OutputFile,
                          ScanInstance.getFrontendOpts().Inputs)};
    if (Format == ScanningOutputFormat::Make) {
      // Only 'Make' scanning needs to force this because that mode depends on
      // getting the dependencies directly from \p DependencyFileGenerator.
      Opts->IncludeSystemHeaders = true;
    }

    auto reportError = [&ScanInstance](Error &&E) -> bool {
      ScanInstance.getDiagnostics().Report(diag::err_cas_depscan_failed)
          << std::move(E);
      return false;
    };

    // FIXME: The caller APIs in \p DependencyScanningTool expect a specific
    // DependencyCollector to get attached to the preprocessor in order to
    // function properly (e.g. \p FullDependencyConsumer needs \p
    // ModuleDepCollector) but this association is very indirect via the value
    // of the \p ScanningOutputFormat. We should remove \p Format field from
    // \p DependencyScanningAction, and have the callers pass in a
    // “DependencyCollector factory” so the connection of collector<->consumer
    // is explicit in each \p DependencyScanningTool function.
    switch (Format) {
    case ScanningOutputFormat::Make:
    case ScanningOutputFormat::Tree:
      ScanInstance.addDependencyCollector(
          std::make_shared<DependencyConsumerForwarder>(
              std::move(Opts), WorkingDirectory, Consumer, EmitDependencyFile));
      break;
    case ScanningOutputFormat::IncludeTree: {
      auto &IncConsumer = static_cast<PPIncludeActionsConsumer &>(Consumer);
      auto IncTreeCollector = std::make_shared<IncludeTreeCollector>(
          IncConsumer, std::move(Opts), EmitDependencyFile);
      if (Error E = IncTreeCollector->initialize(
              ScanInstance.getInvocation(), IncConsumer.getPrefixMapping()))
        return reportError(std::move(E));
      ScanInstance.addDependencyCollector(std::move(IncTreeCollector));
      break;
    }
    case ScanningOutputFormat::Full:
    case ScanningOutputFormat::FullTree:
      MDC = std::make_shared<ModuleDepCollector>(
          std::move(Opts), ScanInstance, Consumer, OriginalInvocation,
          WorkingDirectory, OptimizeArgs, EagerLoadModules);
      ScanInstance.addDependencyCollector(MDC);
      if (CacheFS) {
        ScanInstance.setGenModuleActionWrapper(
            [CacheFS = CacheFS](const FrontendOptions &Opts,
                                std::unique_ptr<FrontendAction> Wrapped) {
              return std::make_unique<WrapScanModuleBuildAction>(
                  std::move(Wrapped), std::move(CacheFS));
            });
      }
      break;
    }

    // Consider different header search and diagnostic options to create
    // different modules. This avoids the unsound aliasing of module PCMs.
    //
    // TODO: Implement diagnostic bucketing to reduce the impact of strict
    // context hashing.
    ScanInstance.getHeaderSearchOpts().ModulesStrictContextHash = true;

    std::unique_ptr<FrontendAction> Action;

    if (ModuleName)
      Action = std::make_unique<GetDependenciesByModuleNameAction>(*ModuleName);
    else
      Action = std::make_unique<ReadPCHAndPreprocessAction>();

    if (Error E = Consumer.initialize(ScanInstance))
      return reportError(std::move(E));

    const bool Result = ScanInstance.ExecuteAction(*Action);

    if (Error E = Consumer.finalize(ScanInstance))
      return reportError(std::move(E));

    if (CacheFS) {
      // Exclude the module cache from tracking. The implicit build pcms should
      // not be needed after scanning.
      if (!ScanInstance.getHeaderSearchOpts().ModuleCachePath.empty())
        (void)CacheFS->excludeFromTracking(
            ScanInstance.getHeaderSearchOpts().ModuleCachePath);

      // Handle profile mappings.
      (void)CacheFS->status(
          OriginalInvocation.getCodeGenOpts().ProfileInstrumentUsePath);
      (void)CacheFS->status(
          OriginalInvocation.getCodeGenOpts().SampleProfileFile);
      (void)CacheFS->status(
          OriginalInvocation.getCodeGenOpts().ProfileRemappingFile);

      auto Tree = CacheFS->createTreeFromNewAccesses(RemapPath);
      if (Tree) {
        if (MDC)
          MDC->setMainFileCASFileSystemRootID(Tree->getID());
        Consumer.handleCASFileSystemRootID(Tree->getID());
      } else
        ScanInstance.getDiagnostics().Report(diag::err_cas_depscan_failed)
            << Tree.takeError();
    }

    if (Result)
      setLastCC1Arguments(std::move(OriginalInvocation));

    return Result;
  }

  bool hasScanned() const { return Scanned; }

  /// Take the cc1 arguments corresponding to the most recent invocation used
  /// with this action. Any modifications implied by the discovered dependencies
  /// will have already been applied.
  std::vector<std::string> takeLastCC1Arguments() {
    std::vector<std::string> Result;
    std::swap(Result, LastCC1Arguments); // Reset LastCC1Arguments to empty.
    return Result;
  }

  IntrusiveRefCntPtr<llvm::vfs::FileSystem> getDepScanFS() {
    if (DepFS) {
      assert(!DepCASFS && "CAS DepFS should not be set");
      return DepFS;
    }
    if (DepCASFS) {
      assert(!DepFS && "DepFS should not be set");
      return DepCASFS;
    }
    return nullptr;
  }

private:
  void setLastCC1Arguments(CompilerInvocation &&CI) {
    if (MDC)
      MDC->applyDiscoveredDependencies(CI);
    LastCC1Arguments = CI.getCC1CommandLine();
  }

private:
  StringRef WorkingDirectory;
  DependencyConsumer &Consumer;
  llvm::IntrusiveRefCntPtr<DependencyScanningWorkerFilesystem> DepFS;
  llvm::IntrusiveRefCntPtr<DependencyScanningCASFilesystem> DepCASFS;
  llvm::IntrusiveRefCntPtr<llvm::cas::CachingOnDiskFileSystem> CacheFS;
  ScanningOutputFormat Format;
  bool OptimizeArgs;
  bool EagerLoadModules;
  bool DisableFree;
  const CASOptions &CASOpts;
  RemapPathCallback RemapPath;
  bool EmitDependencyFile = false;
  bool DiagGenerationAsCompilation;
  Optional<StringRef> ModuleName;
  Optional<CompilerInstance> ScanInstanceStorage;
  std::shared_ptr<ModuleDepCollector> MDC;
  std::vector<std::string> LastCC1Arguments;
  bool Scanned = false;
  raw_ostream *VerboseOS;
};

} // end anonymous namespace

DependencyScanningWorker::DependencyScanningWorker(
    DependencyScanningService &Service,
    llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS)
    : Format(Service.getFormat()), OptimizeArgs(Service.canOptimizeArgs()),
      CASOpts(Service.getCASOpts()), UseCAS(Service.useCASScanning()),
      EagerLoadModules(Service.shouldEagerLoadModules()) {
  PCHContainerOps = std::make_shared<PCHContainerOperations>();
  PCHContainerOps->registerReader(
      std::make_unique<ObjectFilePCHContainerReader>());
  // We don't need to write object files, but the current PCH implementation
  // requires the writer to be registered as well.
  PCHContainerOps->registerWriter(
      std::make_unique<ObjectFilePCHContainerWriter>());

  if (Service.useCASScanning()) {
    CacheFS = Service.getSharedFS().createProxyFS();
    DepCASFS = new DependencyScanningCASFilesystem(CacheFS, Service.getCache());
    BaseFS = DepCASFS;
    return;
  }

  switch (Service.getMode()) {
  case ScanningMode::DependencyDirectivesScan:
    DepFS =
        new DependencyScanningWorkerFilesystem(Service.getSharedCache(), FS);
    BaseFS = DepFS;
    break;
  case ScanningMode::CanonicalPreprocessing:
    DepFS = nullptr;
    BaseFS = FS;
    break;
  }
}

llvm::IntrusiveRefCntPtr<FileManager>
DependencyScanningWorker::getOrCreateFileManager() const {
  return new FileManager(FileSystemOptions(), BaseFS);
}

llvm::Error DependencyScanningWorker::computeDependencies(
    StringRef WorkingDirectory, const std::vector<std::string> &CommandLine,
    DependencyConsumer &Consumer, llvm::Optional<StringRef> ModuleName) {
  std::vector<const char *> CLI;
  for (const std::string &Arg : CommandLine)
    CLI.push_back(Arg.c_str());
  auto DiagOpts = CreateAndPopulateDiagOpts(CLI);
  sanitizeDiagOpts(*DiagOpts);

  // Capture the emitted diagnostics and report them to the client
  // in the case of a failure.
  std::string DiagnosticOutput;
  llvm::raw_string_ostream DiagnosticsOS(DiagnosticOutput);
  TextDiagnosticPrinter DiagPrinter(DiagnosticsOS, DiagOpts.release());

  if (computeDependencies(WorkingDirectory, CommandLine, Consumer, DiagPrinter,
                          ModuleName))
    return llvm::Error::success();
  return llvm::make_error<llvm::StringError>(DiagnosticsOS.str(),
                                             llvm::inconvertibleErrorCode());
}

static bool forEachDriverJob(
    ArrayRef<std::string> Args, DiagnosticsEngine &Diags, FileManager &FM,
    llvm::function_ref<bool(const driver::Command &Cmd)> Callback) {
  std::unique_ptr<driver::Driver> Driver = std::make_unique<driver::Driver>(
      Args[0], llvm::sys::getDefaultTargetTriple(), Diags,
      "clang LLVM compiler", &FM.getVirtualFileSystem());
  Driver->setTitle("clang_based_tool");

  std::vector<const char *> Argv;
  for (const std::string &Arg : Args)
    Argv.push_back(Arg.c_str());

  const std::unique_ptr<driver::Compilation> Compilation(
      Driver->BuildCompilation(llvm::makeArrayRef(Argv)));
  if (!Compilation)
    return false;

  for (const driver::Command &Job : Compilation->getJobs()) {
    if (!Callback(Job))
      return false;
  }
  return true;
}

bool DependencyScanningWorker::computeDependencies(
    StringRef WorkingDirectory, const std::vector<std::string> &CommandLine,
    DependencyConsumer &Consumer, DiagnosticConsumer &DC,
    llvm::Optional<StringRef> ModuleName) {
  // Reset what might have been modified in the previous worker invocation.
  BaseFS->setCurrentWorkingDirectory(WorkingDirectory);

  Optional<std::vector<std::string>> ModifiedCommandLine;
  llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> ModifiedFS;
  if (ModuleName) {
    ModifiedCommandLine = CommandLine;
    ModifiedCommandLine->emplace_back(*ModuleName);

    auto OverlayFS =
        llvm::makeIntrusiveRefCnt<llvm::vfs::OverlayFileSystem>(BaseFS);
    auto InMemoryFS =
        llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
    InMemoryFS->setCurrentWorkingDirectory(WorkingDirectory);
    InMemoryFS->addFile(*ModuleName, 0, llvm::MemoryBuffer::getMemBuffer(""));
    OverlayFS->pushOverlay(InMemoryFS);
    ModifiedFS = OverlayFS;
  }

  const std::vector<std::string> &FinalCommandLine =
      ModifiedCommandLine ? *ModifiedCommandLine : CommandLine;

  FileSystemOptions FSOpts;
  FSOpts.WorkingDir = WorkingDirectory.str();
  auto FileMgr = llvm::makeIntrusiveRefCnt<FileManager>(
      FSOpts, ModifiedFS ? ModifiedFS : BaseFS);

  std::vector<const char *> FinalCCommandLine(CommandLine.size(), nullptr);
  llvm::transform(CommandLine, FinalCCommandLine.begin(),
                  [](const std::string &Str) { return Str.c_str(); });

  auto DiagOpts = CreateAndPopulateDiagOpts(FinalCCommandLine);
  sanitizeDiagOpts(*DiagOpts);
  IntrusiveRefCntPtr<DiagnosticsEngine> Diags =
      CompilerInstance::createDiagnostics(DiagOpts.release(), &DC,
                                          /*ShouldOwnClient=*/false);

  // Although `Diagnostics` are used only for command-line parsing, the
  // custom `DiagConsumer` might expect a `SourceManager` to be present.
  SourceManager SrcMgr(*Diags, *FileMgr);
  Diags->setSourceManager(&SrcMgr);
  // DisableFree is modified by Tooling for running
  // in-process; preserve the original value, which is
  // always true for a driver invocation.
  bool DisableFree = true;
  DependencyScanningAction Action(
      WorkingDirectory, Consumer, DepFS, DepCASFS, CacheFS, Format,
      OptimizeArgs, EagerLoadModules, DisableFree,
      /*EmitDependencyFile=*/false,
      /*DiagGenerationAsCompilation=*/false, getCASOpts(),
      /*RemapPath=*/nullptr, ModuleName);
  bool Success = forEachDriverJob(
      FinalCommandLine, *Diags, *FileMgr, [&](const driver::Command &Cmd) {
        if (StringRef(Cmd.getCreator().getName()) != "clang") {
          // Non-clang command. Just pass through to the dependency
          // consumer.
          Consumer.handleBuildCommand(
              {Cmd.getExecutable(),
               {Cmd.getArguments().begin(), Cmd.getArguments().end()}});
          return true;
        }

        std::vector<std::string> Argv;
        Argv.push_back(Cmd.getExecutable());
        Argv.insert(Argv.end(), Cmd.getArguments().begin(),
                    Cmd.getArguments().end());

        // Create an invocation that uses the underlying file
        // system to ensure that any file system requests that
        // are made by the driver do not go through the
        // dependency scanning filesystem.
        ToolInvocation Invocation(std::move(Argv), &Action, &*FileMgr,
                                  PCHContainerOps);
        Invocation.setDiagnosticConsumer(Diags->getClient());
        Invocation.setDiagnosticOptions(&Diags->getDiagnosticOptions());
        if (!Invocation.run())
          return false;

        std::vector<std::string> Args = Action.takeLastCC1Arguments();
        Consumer.handleBuildCommand({Cmd.getExecutable(), std::move(Args)});
        return true;
      });

  if (Success && !Action.hasScanned())
    Diags->Report(diag::err_fe_expected_compiler_job)
        << llvm::join(FinalCommandLine, " ");
  return Success && Action.hasScanned();
}

void DependencyScanningWorker::computeDependenciesFromCompilerInvocation(
    std::shared_ptr<CompilerInvocation> Invocation, StringRef WorkingDirectory,
    DependencyConsumer &DepsConsumer, RemapPathCallback RemapPath,
    DiagnosticConsumer &DiagsConsumer, raw_ostream *VerboseOS,
    bool DiagGenerationAsCompilation) {
  BaseFS->setCurrentWorkingDirectory(WorkingDirectory);

  // Adjust the invocation.
  auto &Frontend = Invocation->getFrontendOpts();
  Frontend.ProgramAction = frontend::RunPreprocessorOnly;
  Frontend.OutputFile = "/dev/null";
  Frontend.DisableFree = false;

  // // Reset dependency options.
  // Dependencies = DependencyOutputOptions();
  // Dependencies.IncludeSystemHeaders = true;
  // Dependencies.OutputFile = "/dev/null";

  // Make the output file path absolute relative to WorkingDirectory.
  std::string &DepFile = Invocation->getDependencyOutputOpts().OutputFile;
  if (!DepFile.empty() && !llvm::sys::path::is_absolute(DepFile)) {
    // FIXME: On Windows, WorkingDirectory is insufficient for making an
    // absolute path if OutputFile has a root name.
    llvm::SmallString<128> Path = StringRef(DepFile);
    llvm::sys::fs::make_absolute(WorkingDirectory, Path);
    DepFile = Path.str().str();
  }

  // FIXME: EmitDependencyFile should only be set when it's for a real
  // compilation.
  DependencyScanningAction Action(
      WorkingDirectory, DepsConsumer, DepFS, DepCASFS, CacheFS, Format,
      /*OptimizeArgs=*/false, /*DisableFree=*/false, EagerLoadModules,
      /*EmitDependencyFile=*/!DepFile.empty(), DiagGenerationAsCompilation,
      getCASOpts(), RemapPath,
      /*ModuleName=*/None, VerboseOS);

  // Ignore result; we're just collecting dependencies.
  //
  // FIXME: will clients other than -cc1scand care?
  IntrusiveRefCntPtr<FileManager> ActiveFiles =
      new FileManager(Invocation->getFileSystemOpts(), BaseFS);
  (void)Action.runInvocation(std::move(Invocation), ActiveFiles.get(),
                             PCHContainerOps, &DiagsConsumer);
}
