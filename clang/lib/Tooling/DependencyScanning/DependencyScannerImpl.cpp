//===- DependencyScanner.cpp - Performs module dependency scanning --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DependencyScannerImpl.h"
#include "clang/Basic/DiagnosticFrontend.h"
#include "clang/Basic/DiagnosticSerialization.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/DependencyScanning/DependencyScanningWorker.h"

using namespace clang;
using namespace tooling;
using namespace dependencies;

namespace {
/// Forwards the gatherered dependencies to the consumer.
class DependencyConsumerForwarder : public DependencyFileGenerator {
public:
  DependencyConsumerForwarder(std::unique_ptr<DependencyOutputOptions> Opts,
                              StringRef WorkingDirectory, DependencyConsumer &C)
      : DependencyFileGenerator(*Opts), WorkingDirectory(WorkingDirectory),
        Opts(std::move(Opts)), C(C) {}

  void finishedMainFile(DiagnosticsEngine &Diags) override {
    C.handleDependencyOutputOpts(*Opts);
    llvm::SmallString<256> CanonPath;
    for (const auto &File : getDependencies()) {
      CanonPath = File;
      llvm::sys::path::remove_dots(CanonPath, /*remove_dot_dot=*/true);
      llvm::sys::fs::make_absolute(WorkingDirectory, CanonPath);
      C.handleFileDependency(CanonPath);
    }
  }

private:
  StringRef WorkingDirectory;
  std::unique_ptr<DependencyOutputOptions> Opts;
  DependencyConsumer &C;
};

static bool checkHeaderSearchPaths(const HeaderSearchOptions &HSOpts,
                                   const HeaderSearchOptions &ExistingHSOpts,
                                   DiagnosticsEngine *Diags,
                                   const LangOptions &LangOpts) {
  if (LangOpts.Modules) {
    if (HSOpts.VFSOverlayFiles != ExistingHSOpts.VFSOverlayFiles) {
      if (Diags) {
        Diags->Report(diag::warn_pch_vfsoverlay_mismatch);
        auto VFSNote = [&](int Type, ArrayRef<std::string> VFSOverlays) {
          if (VFSOverlays.empty()) {
            Diags->Report(diag::note_pch_vfsoverlay_empty) << Type;
          } else {
            std::string Files = llvm::join(VFSOverlays, "\n");
            Diags->Report(diag::note_pch_vfsoverlay_files) << Type << Files;
          }
        };
        VFSNote(0, HSOpts.VFSOverlayFiles);
        VFSNote(1, ExistingHSOpts.VFSOverlayFiles);
      }
    }
  }
  return false;
}

using PrebuiltModuleFilesT = decltype(HeaderSearchOptions::PrebuiltModuleFiles);

/// A listener that collects the imported modules and the input
/// files. While visiting, collect vfsoverlays and file inputs that determine
/// whether prebuilt modules fully resolve in stable directories.
class PrebuiltModuleListener : public ASTReaderListener {
public:
  PrebuiltModuleListener(PrebuiltModuleFilesT &PrebuiltModuleFiles,
                         llvm::SmallVector<std::string> &NewModuleFiles,
                         PrebuiltModulesAttrsMap &PrebuiltModulesASTMap,
                         const HeaderSearchOptions &HSOpts,
                         const LangOptions &LangOpts, DiagnosticsEngine &Diags,
                         const ArrayRef<StringRef> StableDirs)
      : PrebuiltModuleFiles(PrebuiltModuleFiles),
        NewModuleFiles(NewModuleFiles),
        PrebuiltModulesASTMap(PrebuiltModulesASTMap), ExistingHSOpts(HSOpts),
        ExistingLangOpts(LangOpts), Diags(Diags), StableDirs(StableDirs) {}

  bool needsImportVisitation() const override { return true; }
  bool needsInputFileVisitation() override { return true; }
  bool needsSystemInputFileVisitation() override { return true; }

  /// Accumulate the modules are transitively depended on by the initial
  /// prebuilt module.
  void visitImport(StringRef ModuleName, StringRef Filename) override {
    if (PrebuiltModuleFiles.insert({ModuleName.str(), Filename.str()}).second)
      NewModuleFiles.push_back(Filename.str());

    auto PrebuiltMapEntry = PrebuiltModulesASTMap.try_emplace(Filename);
    PrebuiltModuleASTAttrs &PrebuiltModule = PrebuiltMapEntry.first->second;
    if (PrebuiltMapEntry.second)
      PrebuiltModule.setInStableDir(!StableDirs.empty());

    if (auto It = PrebuiltModulesASTMap.find(CurrentFile);
        It != PrebuiltModulesASTMap.end() && CurrentFile != Filename)
      PrebuiltModule.addDependent(It->getKey());
  }

  /// For each input file discovered, check whether it's external path is in a
  /// stable directory. Traversal is stopped if the current module is not
  /// considered stable.
  bool visitInputFileAsRequested(StringRef FilenameAsRequested,
                                 StringRef Filename, bool isSystem,
                                 bool isOverridden,
                                 bool isExplicitModule) override {
    if (StableDirs.empty())
      return false;
    auto PrebuiltEntryIt = PrebuiltModulesASTMap.find(CurrentFile);
    if ((PrebuiltEntryIt == PrebuiltModulesASTMap.end()) ||
        (!PrebuiltEntryIt->second.isInStableDir()))
      return false;

    PrebuiltEntryIt->second.setInStableDir(
        isPathInStableDir(StableDirs, Filename));
    return PrebuiltEntryIt->second.isInStableDir();
  }

  /// Update which module that is being actively traversed.
  void visitModuleFile(StringRef Filename,
                       serialization::ModuleKind Kind) override {
    // If the CurrentFile is not
    // considered stable, update any of it's transitive dependents.
    auto PrebuiltEntryIt = PrebuiltModulesASTMap.find(CurrentFile);
    if ((PrebuiltEntryIt != PrebuiltModulesASTMap.end()) &&
        !PrebuiltEntryIt->second.isInStableDir())
      PrebuiltEntryIt->second.updateDependentsNotInStableDirs(
          PrebuiltModulesASTMap);
    CurrentFile = Filename;
  }

  /// Check the header search options for a given module when considering
  /// if the module comes from stable directories.
  bool ReadHeaderSearchOptions(const HeaderSearchOptions &HSOpts,
                               StringRef ModuleFilename,
                               StringRef SpecificModuleCachePath,
                               bool Complain) override {

    auto PrebuiltMapEntry = PrebuiltModulesASTMap.try_emplace(CurrentFile);
    PrebuiltModuleASTAttrs &PrebuiltModule = PrebuiltMapEntry.first->second;
    if (PrebuiltMapEntry.second)
      PrebuiltModule.setInStableDir(!StableDirs.empty());

    if (PrebuiltModule.isInStableDir())
      PrebuiltModule.setInStableDir(areOptionsInStableDir(StableDirs, HSOpts));

    return false;
  }

  /// Accumulate vfsoverlays used to build these prebuilt modules.
  bool ReadHeaderSearchPaths(const HeaderSearchOptions &HSOpts,
                             bool Complain) override {

    auto PrebuiltMapEntry = PrebuiltModulesASTMap.try_emplace(CurrentFile);
    PrebuiltModuleASTAttrs &PrebuiltModule = PrebuiltMapEntry.first->second;
    if (PrebuiltMapEntry.second)
      PrebuiltModule.setInStableDir(!StableDirs.empty());

    PrebuiltModule.setVFS(
        llvm::StringSet<>(llvm::from_range, HSOpts.VFSOverlayFiles));

    return checkHeaderSearchPaths(
        HSOpts, ExistingHSOpts, Complain ? &Diags : nullptr, ExistingLangOpts);
  }

private:
  PrebuiltModuleFilesT &PrebuiltModuleFiles;
  llvm::SmallVector<std::string> &NewModuleFiles;
  PrebuiltModulesAttrsMap &PrebuiltModulesASTMap;
  const HeaderSearchOptions &ExistingHSOpts;
  const LangOptions &ExistingLangOpts;
  DiagnosticsEngine &Diags;
  std::string CurrentFile;
  const ArrayRef<StringRef> StableDirs;
};

/// Visit the given prebuilt module and collect all of the modules it
/// transitively imports and contributing input files.
static bool visitPrebuiltModule(StringRef PrebuiltModuleFilename,
                                CompilerInstance &CI,
                                PrebuiltModuleFilesT &ModuleFiles,
                                PrebuiltModulesAttrsMap &PrebuiltModulesASTMap,
                                DiagnosticsEngine &Diags,
                                const ArrayRef<StringRef> StableDirs) {
  // List of module files to be processed.
  llvm::SmallVector<std::string> Worklist;

  PrebuiltModuleListener Listener(ModuleFiles, Worklist, PrebuiltModulesASTMap,
                                  CI.getHeaderSearchOpts(), CI.getLangOpts(),
                                  Diags, StableDirs);

  Listener.visitModuleFile(PrebuiltModuleFilename,
                           serialization::MK_ExplicitModule);
  if (ASTReader::readASTFileControlBlock(
          PrebuiltModuleFilename, CI.getFileManager(), CI.getModuleCache(),
          CI.getPCHContainerReader(),
          /*FindModuleFileExtensions=*/false, Listener,
          /*ValidateDiagnosticOptions=*/false, ASTReader::ARR_OutOfDate))
    return true;

  while (!Worklist.empty()) {
    Listener.visitModuleFile(Worklist.back(), serialization::MK_ExplicitModule);
    if (ASTReader::readASTFileControlBlock(
            Worklist.pop_back_val(), CI.getFileManager(), CI.getModuleCache(),
            CI.getPCHContainerReader(),
            /*FindModuleFileExtensions=*/false, Listener,
            /*ValidateDiagnosticOptions=*/false))
      return true;
  }
  return false;
}

/// Transform arbitrary file name into an object-like file name.
static std::string makeObjFileName(StringRef FileName) {
  SmallString<128> ObjFileName(FileName);
  llvm::sys::path::replace_extension(ObjFileName, "o");
  return std::string(ObjFileName);
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

// Clang implements -D and -U by splatting text into a predefines buffer. This
// allows constructs such as `-DFඞ=3 "-D F\u{0D9E} 4 3 2”` to be accepted and
// define the same macro, or adding C++ style comments before the macro name.
//
// This function checks that the first non-space characters in the macro
// obviously form an identifier that can be uniqued on without lexing. Failing
// to do this could lead to changing the final definition of a macro.
//
// We could set up a preprocessor and actually lex the name, but that's very
// heavyweight for a situation that will almost never happen in practice.
static std::optional<StringRef> getSimpleMacroName(StringRef Macro) {
  StringRef Name = Macro.split("=").first.ltrim(" \t");
  std::size_t I = 0;

  auto FinishName = [&]() -> std::optional<StringRef> {
    StringRef SimpleName = Name.slice(0, I);
    if (SimpleName.empty())
      return std::nullopt;
    return SimpleName;
  };

  for (; I != Name.size(); ++I) {
    switch (Name[I]) {
    case '(': // Start of macro parameter list
    case ' ': // End of macro name
    case '\t':
      return FinishName();
    case '_':
      continue;
    default:
      if (llvm::isAlnum(Name[I]))
        continue;
      return std::nullopt;
    }
  }
  return FinishName();
}

static void canonicalizeDefines(PreprocessorOptions &PPOpts) {
  using MacroOpt = std::pair<StringRef, std::size_t>;
  std::vector<MacroOpt> SimpleNames;
  SimpleNames.reserve(PPOpts.Macros.size());
  std::size_t Index = 0;
  for (const auto &M : PPOpts.Macros) {
    auto SName = getSimpleMacroName(M.first);
    // Skip optimizing if we can't guarantee we can preserve relative order.
    if (!SName)
      return;
    SimpleNames.emplace_back(*SName, Index);
    ++Index;
  }

  llvm::stable_sort(SimpleNames, llvm::less_first());
  // Keep the last instance of each macro name by going in reverse
  auto NewEnd = std::unique(
      SimpleNames.rbegin(), SimpleNames.rend(),
      [](const MacroOpt &A, const MacroOpt &B) { return A.first == B.first; });
  SimpleNames.erase(SimpleNames.begin(), NewEnd.base());

  // Apply permutation.
  decltype(PPOpts.Macros) NewMacros;
  NewMacros.reserve(SimpleNames.size());
  for (std::size_t I = 0, E = SimpleNames.size(); I != E; ++I) {
    std::size_t OriginalIndex = SimpleNames[I].second;
    // We still emit undefines here as they may be undefining a predefined macro
    NewMacros.push_back(std::move(PPOpts.Macros[OriginalIndex]));
  }
  std::swap(PPOpts.Macros, NewMacros);
}

class ScanningDependencyDirectivesGetter : public DependencyDirectivesGetter {
  DependencyScanningWorkerFilesystem *DepFS;

public:
  ScanningDependencyDirectivesGetter(FileManager &FileMgr) : DepFS(nullptr) {
    FileMgr.getVirtualFileSystem().visit([&](llvm::vfs::FileSystem &FS) {
      auto *DFS = llvm::dyn_cast<DependencyScanningWorkerFilesystem>(&FS);
      if (DFS) {
        assert(!DepFS && "Found multiple scanning VFSs");
        DepFS = DFS;
      }
    });
    assert(DepFS && "Did not find scanning VFS");
  }

  std::unique_ptr<DependencyDirectivesGetter>
  cloneFor(FileManager &FileMgr) override {
    return std::make_unique<ScanningDependencyDirectivesGetter>(FileMgr);
  }

  std::optional<ArrayRef<dependency_directives_scan::Directive>>
  operator()(FileEntryRef File) override {
    return DepFS->getDirectiveTokens(File.getName());
  }
};
} // namespace

/// Sanitize diagnostic options for dependency scan.
void clang::tooling::dependencies::sanitizeDiagOpts(
    DiagnosticOptions &DiagOpts) {
  // Don't print 'X warnings and Y errors generated'.
  DiagOpts.ShowCarets = false;
  // Don't write out diagnostic file.
  DiagOpts.DiagnosticSerializationFile.clear();
  // Don't emit warnings except for scanning specific warnings.
  // TODO: It would be useful to add a more principled way to ignore all
  //       warnings that come from source code. The issue is that we need to
  //       ignore warnings that could be surpressed by
  //       `#pragma clang diagnostic`, while still allowing some scanning
  //       warnings for things we're not ready to turn into errors yet.
  //       See `test/ClangScanDeps/diagnostic-pragmas.c` for an example.
  llvm::erase_if(DiagOpts.Warnings, [](StringRef Warning) {
    return llvm::StringSwitch<bool>(Warning)
        .Cases("pch-vfs-diff", "error=pch-vfs-diff", false)
        .StartsWith("no-error=", false)
        .Default(true);
  });
}

bool DependencyScanningAction::runInvocation(
    std::shared_ptr<CompilerInvocation> Invocation,
    IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS,
    std::shared_ptr<PCHContainerOperations> PCHContainerOps,
    DiagnosticConsumer *DiagConsumer) {
  // Making sure that we canonicalize the defines before we create the deep
  // copy to avoid unnecessary variants in the scanner and in the resulting
  // explicit command lines.
  if (any(Service.getOptimizeArgs() & ScanningOptimizations::Macros))
    canonicalizeDefines(Invocation->getPreprocessorOpts());

  // Make a deep copy of the original Clang invocation.
  CompilerInvocation OriginalInvocation(*Invocation);

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
  auto ModCache = makeInProcessModuleCache(Service.getModuleCacheEntries());
  ScanInstanceStorage.emplace(std::move(Invocation), std::move(PCHContainerOps),
                              ModCache.get());
  CompilerInstance &ScanInstance = *ScanInstanceStorage;
  ScanInstance.setBuildingModule(false);

  ScanInstance.createVirtualFileSystem(FS, DiagConsumer);

  // Create the compiler's actual diagnostics engine.
  sanitizeDiagOpts(ScanInstance.getDiagnosticOpts());
  assert(!DiagConsumerFinished && "attempt to reuse finished consumer");
  ScanInstance.createDiagnostics(DiagConsumer, /*ShouldOwnClient=*/false);
  if (!ScanInstance.hasDiagnostics())
    return false;

  ScanInstance.getPreprocessorOpts().AllowPCHWithDifferentModulesCachePath =
      true;

  if (ScanInstance.getHeaderSearchOpts().ModulesValidateOncePerBuildSession)
    ScanInstance.getHeaderSearchOpts().BuildSessionTimestamp =
        Service.getBuildSessionTimestamp();

  ScanInstance.getFrontendOpts().DisableFree = false;
  ScanInstance.getFrontendOpts().GenerateGlobalModuleIndex = false;
  ScanInstance.getFrontendOpts().UseGlobalModuleIndex = false;
  // This will prevent us compiling individual modules asynchronously since
  // FileManager is not thread-safe, but it does improve performance for now.
  ScanInstance.getFrontendOpts().ModulesShareFileManager = true;
  ScanInstance.getHeaderSearchOpts().ModuleFormat = "raw";
  ScanInstance.getHeaderSearchOpts().ModulesIncludeVFSUsage =
      any(Service.getOptimizeArgs() & ScanningOptimizations::VFS);

  // Create a new FileManager to match the invocation's FileSystemOptions.
  auto *FileMgr = ScanInstance.createFileManager();

  // Use the dependency scanning optimized file system if requested to do so.
  if (DepFS) {
    DepFS->resetBypassedPathPrefix();
    if (!ScanInstance.getHeaderSearchOpts().ModuleCachePath.empty()) {
      SmallString<256> ModulesCachePath;
      normalizeModuleCachePath(
          *FileMgr, ScanInstance.getHeaderSearchOpts().ModuleCachePath,
          ModulesCachePath);
      DepFS->setBypassedPathPrefix(ModulesCachePath);
    }

    ScanInstance.setDependencyDirectivesGetter(
        std::make_unique<ScanningDependencyDirectivesGetter>(*FileMgr));
  }

  ScanInstance.createSourceManager(*FileMgr);

  // Create a collection of stable directories derived from the ScanInstance
  // for determining whether module dependencies would fully resolve from
  // those directories.
  llvm::SmallVector<StringRef> StableDirs;
  const StringRef Sysroot = ScanInstance.getHeaderSearchOpts().Sysroot;
  if (!Sysroot.empty() && (llvm::sys::path::root_directory(Sysroot) != Sysroot))
    StableDirs = {Sysroot, ScanInstance.getHeaderSearchOpts().ResourceDir};

  // Store a mapping of prebuilt module files and their properties like header
  // search options. This will prevent the implicit build to create duplicate
  // modules and will force reuse of the existing prebuilt module files
  // instead.
  PrebuiltModulesAttrsMap PrebuiltModulesASTMap;

  if (!ScanInstance.getPreprocessorOpts().ImplicitPCHInclude.empty())
    if (visitPrebuiltModule(
            ScanInstance.getPreprocessorOpts().ImplicitPCHInclude, ScanInstance,
            ScanInstance.getHeaderSearchOpts().PrebuiltModuleFiles,
            PrebuiltModulesASTMap, ScanInstance.getDiagnostics(), StableDirs))
      return false;

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
    Opts->Targets = {deduceDepTarget(ScanInstance.getFrontendOpts().OutputFile,
                                     ScanInstance.getFrontendOpts().Inputs)};
  Opts->IncludeSystemHeaders = true;

  switch (Service.getFormat()) {
  case ScanningOutputFormat::Make:
    ScanInstance.addDependencyCollector(
        std::make_shared<DependencyConsumerForwarder>(
            std::move(Opts), WorkingDirectory, Consumer));
    break;
  case ScanningOutputFormat::P1689:
  case ScanningOutputFormat::Full:
    MDC = std::make_shared<ModuleDepCollector>(
        Service, std::move(Opts), ScanInstance, Consumer, Controller,
        OriginalInvocation, std::move(PrebuiltModulesASTMap), StableDirs);
    ScanInstance.addDependencyCollector(MDC);
    break;
  }

  // Consider different header search and diagnostic options to create
  // different modules. This avoids the unsound aliasing of module PCMs.
  //
  // TODO: Implement diagnostic bucketing to reduce the impact of strict
  // context hashing.
  ScanInstance.getHeaderSearchOpts().ModulesStrictContextHash = true;
  ScanInstance.getHeaderSearchOpts().ModulesSerializeOnlyPreprocessor = true;
  ScanInstance.getHeaderSearchOpts().ModulesSkipDiagnosticOptions = true;
  ScanInstance.getHeaderSearchOpts().ModulesSkipHeaderSearchPaths = true;
  ScanInstance.getHeaderSearchOpts().ModulesSkipPragmaDiagnosticMappings = true;
  ScanInstance.getHeaderSearchOpts().ModulesForceValidateUserHeaders = false;

  // Avoid some checks and module map parsing when loading PCM files.
  ScanInstance.getPreprocessorOpts().ModulesCheckRelocated = false;

  std::unique_ptr<FrontendAction> Action;

  if (Service.getFormat() == ScanningOutputFormat::P1689)
    Action = std::make_unique<PreprocessOnlyAction>();
  else if (ModuleName)
    Action = std::make_unique<GetDependenciesByModuleNameAction>(*ModuleName);
  else
    Action = std::make_unique<ReadPCHAndPreprocessAction>();

  if (ScanInstance.getDiagnostics().hasErrorOccurred())
    return false;

  const bool Result = ScanInstance.ExecuteAction(*Action);

  // ExecuteAction is responsible for calling finish.
  DiagConsumerFinished = true;

  if (Result)
    setLastCC1Arguments(std::move(OriginalInvocation));

  return Result;
}
