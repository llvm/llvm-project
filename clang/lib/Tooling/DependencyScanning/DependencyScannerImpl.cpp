//===- DependencyScannerImpl.cpp - Implements module dependency scanning --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DependencyScannerImpl.h"
#include "clang/Basic/DiagnosticFrontend.h"
#include "clang/Basic/DiagnosticSerialization.h"
#include "clang/Driver/Driver.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/DependencyScanning/DependencyScanningWorker.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/TargetParser/Host.h"

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
      llvm::sys::path::make_absolute(WorkingDirectory, CanonPath);
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

/// Sanitize diagnostic options for dependency scan.
void sanitizeDiagOpts(DiagnosticOptions &DiagOpts) {
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
        .Cases({"pch-vfs-diff", "error=pch-vfs-diff"}, false)
        .StartsWith("no-error=", false)
        .Default(true);
  });
}
} // namespace

std::unique_ptr<DiagnosticOptions>
dependencies::createDiagOptions(ArrayRef<std::string> CommandLine) {
  std::vector<const char *> CLI;
  for (const std::string &Arg : CommandLine)
    CLI.push_back(Arg.c_str());
  auto DiagOpts = CreateAndPopulateDiagOpts(CLI);
  sanitizeDiagOpts(*DiagOpts);
  return DiagOpts;
}

DignosticsEngineWithDiagOpts::DignosticsEngineWithDiagOpts(
    ArrayRef<std::string> CommandLine,
    IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS, DiagnosticConsumer &DC) {
  std::vector<const char *> CCommandLine(CommandLine.size(), nullptr);
  llvm::transform(CommandLine, CCommandLine.begin(),
                  [](const std::string &Str) { return Str.c_str(); });
  DiagOpts = CreateAndPopulateDiagOpts(CCommandLine);
  sanitizeDiagOpts(*DiagOpts);
  DiagEngine = CompilerInstance::createDiagnostics(*FS, *DiagOpts, &DC,
                                                   /*ShouldOwnClient=*/false);
}

std::pair<std::unique_ptr<driver::Driver>, std::unique_ptr<driver::Compilation>>
dependencies::buildCompilation(ArrayRef<std::string> ArgStrs,
                               DiagnosticsEngine &Diags,
                               IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS,
                               llvm::BumpPtrAllocator &Alloc) {
  SmallVector<const char *, 256> Argv;
  Argv.reserve(ArgStrs.size());
  for (const std::string &Arg : ArgStrs)
    Argv.push_back(Arg.c_str());

  std::unique_ptr<driver::Driver> Driver = std::make_unique<driver::Driver>(
      Argv[0], llvm::sys::getDefaultTargetTriple(), Diags,
      "clang LLVM compiler", FS);
  Driver->setTitle("clang_based_tool");

  bool CLMode = driver::IsClangCL(
      driver::getDriverMode(Argv[0], ArrayRef(Argv).slice(1)));

  if (llvm::Error E =
          driver::expandResponseFiles(Argv, CLMode, Alloc, FS.get())) {
    Diags.Report(diag::err_drv_expand_response_file)
        << llvm::toString(std::move(E));
    return std::make_pair(nullptr, nullptr);
  }

  std::unique_ptr<driver::Compilation> Compilation(
      Driver->BuildCompilation(Argv));
  if (!Compilation)
    return std::make_pair(nullptr, nullptr);

  if (Compilation->containsError())
    return std::make_pair(nullptr, nullptr);

  return std::make_pair(std::move(Driver), std::move(Compilation));
}

std::unique_ptr<CompilerInvocation>
dependencies::createCompilerInvocation(ArrayRef<std::string> CommandLine,
                                       DiagnosticsEngine &Diags) {
  llvm::opt::ArgStringList Argv;
  for (const std::string &Str : ArrayRef(CommandLine).drop_front())
    Argv.push_back(Str.c_str());

  auto Invocation = std::make_unique<CompilerInvocation>();
  if (!CompilerInvocation::CreateFromArgs(*Invocation, Argv, Diags)) {
    // FIXME: Should we just go on like cc1_main does?
    return nullptr;
  }
  return Invocation;
}

std::pair<IntrusiveRefCntPtr<llvm::vfs::FileSystem>, std::vector<std::string>>
dependencies::initVFSForTUBuferScanning(
    IntrusiveRefCntPtr<llvm::vfs::FileSystem> BaseFS,
    ArrayRef<std::string> CommandLine, StringRef WorkingDirectory,
    llvm::MemoryBufferRef TUBuffer) {
  // Reset what might have been modified in the previous worker invocation.
  BaseFS->setCurrentWorkingDirectory(WorkingDirectory);

  IntrusiveRefCntPtr<llvm::vfs::FileSystem> ModifiedFS;
  auto OverlayFS =
      llvm::makeIntrusiveRefCnt<llvm::vfs::OverlayFileSystem>(BaseFS);
  auto InMemoryFS = llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
  InMemoryFS->setCurrentWorkingDirectory(WorkingDirectory);
  auto InputPath = TUBuffer.getBufferIdentifier();
  InMemoryFS->addFile(
      InputPath, 0, llvm::MemoryBuffer::getMemBufferCopy(TUBuffer.getBuffer()));
  IntrusiveRefCntPtr<llvm::vfs::FileSystem> InMemoryOverlay = InMemoryFS;

  OverlayFS->pushOverlay(InMemoryOverlay);
  ModifiedFS = OverlayFS;
  std::vector<std::string> ModifiedCommandLine(CommandLine);
  ModifiedCommandLine.emplace_back(InputPath);

  return std::make_pair(ModifiedFS, ModifiedCommandLine);
}

std::pair<IntrusiveRefCntPtr<llvm::vfs::OverlayFileSystem>,
          std::vector<std::string>>
dependencies::initVFSForByNameScanning(
    IntrusiveRefCntPtr<llvm::vfs::FileSystem> BaseFS,
    ArrayRef<std::string> CommandLine, StringRef WorkingDirectory,
    StringRef ModuleName) {
  // Reset what might have been modified in the previous worker invocation.
  BaseFS->setCurrentWorkingDirectory(WorkingDirectory);

  // If we're scanning based on a module name alone, we don't expect the client
  // to provide us with an input file. However, the driver really wants to have
  // one. Let's just make it up to make the driver happy.
  auto OverlayFS =
      llvm::makeIntrusiveRefCnt<llvm::vfs::OverlayFileSystem>(BaseFS);
  auto InMemoryFS = llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
  InMemoryFS->setCurrentWorkingDirectory(WorkingDirectory);
  SmallString<128> FakeInputPath;
  // TODO: We should retry the creation if the path already exists.
  llvm::sys::fs::createUniquePath(ModuleName + "-%%%%%%%%.input", FakeInputPath,
                                  /*MakeAbsolute=*/false);
  InMemoryFS->addFile(FakeInputPath, 0, llvm::MemoryBuffer::getMemBuffer(""));
  IntrusiveRefCntPtr<llvm::vfs::FileSystem> InMemoryOverlay = InMemoryFS;
  OverlayFS->pushOverlay(InMemoryOverlay);

  std::vector<std::string> ModifiedCommandLine(CommandLine);
  ModifiedCommandLine.emplace_back(FakeInputPath);

  return std::make_pair(OverlayFS, ModifiedCommandLine);
}

bool dependencies::initializeScanCompilerInstance(
    CompilerInstance &ScanInstance,
    IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS,
    DiagnosticConsumer *DiagConsumer, DependencyScanningService &Service,
    IntrusiveRefCntPtr<DependencyScanningWorkerFilesystem> DepFS) {
  ScanInstance.setBuildingModule(false);

  ScanInstance.createVirtualFileSystem(FS, DiagConsumer);

  // Create the compiler's actual diagnostics engine.
  sanitizeDiagOpts(ScanInstance.getDiagnosticOpts());
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
  ScanInstance.getFrontendOpts().GenReducedBMI = false;
  ScanInstance.getFrontendOpts().ModuleOutputPath.clear();
  // This will prevent us compiling individual modules asynchronously since
  // FileManager is not thread-safe, but it does improve performance for now.
  ScanInstance.getFrontendOpts().ModulesShareFileManager = true;
  ScanInstance.getHeaderSearchOpts().ModuleFormat = "raw";
  ScanInstance.getHeaderSearchOpts().ModulesIncludeVFSUsage =
      any(Service.getOptimizeArgs() & ScanningOptimizations::VFS);

  // Create a new FileManager to match the invocation's FileSystemOptions.
  ScanInstance.createFileManager();

  // Use the dependency scanning optimized file system if requested to do so.
  if (DepFS) {
    DepFS->resetBypassedPathPrefix();
    SmallString<256> ModulesCachePath;
    normalizeModuleCachePath(ScanInstance.getFileManager(),
                             ScanInstance.getHeaderSearchOpts().ModuleCachePath,
                             ModulesCachePath);
    if (!ModulesCachePath.empty())
      DepFS->setBypassedPathPrefix(ModulesCachePath);

    ScanInstance.setDependencyDirectivesGetter(
        std::make_unique<ScanningDependencyDirectivesGetter>(
            ScanInstance.getFileManager()));
  }

  ScanInstance.createSourceManager();

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

  return true;
}

llvm::SmallVector<StringRef>
dependencies::getInitialStableDirs(const CompilerInstance &ScanInstance) {
  // Create a collection of stable directories derived from the ScanInstance
  // for determining whether module dependencies would fully resolve from
  // those directories.
  llvm::SmallVector<StringRef> StableDirs;
  const StringRef Sysroot = ScanInstance.getHeaderSearchOpts().Sysroot;
  if (!Sysroot.empty() && (llvm::sys::path::root_directory(Sysroot) != Sysroot))
    StableDirs = {Sysroot, ScanInstance.getHeaderSearchOpts().ResourceDir};
  return StableDirs;
}

std::optional<PrebuiltModulesAttrsMap>
dependencies::computePrebuiltModulesASTMap(
    CompilerInstance &ScanInstance, llvm::SmallVector<StringRef> &StableDirs) {
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
      return {};

  return PrebuiltModulesASTMap;
}

std::unique_ptr<DependencyOutputOptions>
dependencies::takeAndUpdateDependencyOutputOptionsFrom(
    CompilerInstance &ScanInstance) {
  // This function moves the existing dependency output options from the
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

  return Opts;
}

std::shared_ptr<ModuleDepCollector>
dependencies::initializeScanInstanceDependencyCollector(
    CompilerInstance &ScanInstance,
    std::unique_ptr<DependencyOutputOptions> DepOutputOpts,
    StringRef WorkingDirectory, DependencyConsumer &Consumer,
    DependencyScanningService &Service, CompilerInvocation &Inv,
    DependencyActionController &Controller,
    PrebuiltModulesAttrsMap PrebuiltModulesASTMap,
    llvm::SmallVector<StringRef> &StableDirs) {
  std::shared_ptr<ModuleDepCollector> MDC;
  switch (Service.getFormat()) {
  case ScanningOutputFormat::Make:
    ScanInstance.addDependencyCollector(
        std::make_shared<DependencyConsumerForwarder>(
            std::move(DepOutputOpts), WorkingDirectory, Consumer));
    break;
  case ScanningOutputFormat::P1689:
  case ScanningOutputFormat::Full:
    MDC = std::make_shared<ModuleDepCollector>(
        Service, std::move(DepOutputOpts), ScanInstance, Consumer, Controller,
        Inv, std::move(PrebuiltModulesASTMap), StableDirs);
    ScanInstance.addDependencyCollector(MDC);
    break;
  }

  return MDC;
}

bool DependencyScanningAction::runInvocation(
    std::unique_ptr<CompilerInvocation> Invocation,
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

  assert(!DiagConsumerFinished && "attempt to reuse finished consumer");
  if (!initializeScanCompilerInstance(ScanInstance, FS, DiagConsumer, Service,
                                      DepFS))
    return false;

  llvm::SmallVector<StringRef> StableDirs = getInitialStableDirs(ScanInstance);
  auto MaybePrebuiltModulesASTMap =
      computePrebuiltModulesASTMap(ScanInstance, StableDirs);
  if (!MaybePrebuiltModulesASTMap)
    return false;

  auto DepOutputOpts = takeAndUpdateDependencyOutputOptionsFrom(ScanInstance);

  MDC = initializeScanInstanceDependencyCollector(
      ScanInstance, std::move(DepOutputOpts), WorkingDirectory, Consumer,
      Service, OriginalInvocation, Controller, *MaybePrebuiltModulesASTMap,
      StableDirs);

  std::unique_ptr<FrontendAction> Action;

  if (Service.getFormat() == ScanningOutputFormat::P1689)
    Action = std::make_unique<PreprocessOnlyAction>();
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

bool CompilerInstanceWithContext::initialize(DiagnosticConsumer *DC) {
  if (DC) {
    DiagConsumer = DC;
  } else {
    DiagPrinterWithOS =
        std::make_unique<TextDiagnosticsPrinterWithOutput>(CommandLine);
    DiagConsumer = &DiagPrinterWithOS->DiagPrinter;
  }

  std::tie(OverlayFS, CommandLine) = initVFSForByNameScanning(
      Worker.BaseFS, CommandLine, CWD, "ScanningByName");

  DiagEngineWithCmdAndOpts = std::make_unique<DignosticsEngineWithDiagOpts>(
      CommandLine, OverlayFS, *DiagConsumer);

  std::tie(Driver, Compilation) = buildCompilation(
      CommandLine, *DiagEngineWithCmdAndOpts->DiagEngine, OverlayFS, Alloc);

  if (!Compilation)
    return false;

  assert(Compilation->getJobs().size() &&
         "Must have a job list of non-zero size");
  const driver::Command &Command = *(Compilation->getJobs().begin());
  const auto &CommandArgs = Command.getArguments();
  assert(!CommandArgs.empty() && "Cannot have a command with 0 args");
  assert(StringRef(CommandArgs[0]) == "-cc1" && "Requires a cc1 job.");
  OriginalInvocation = std::make_unique<CompilerInvocation>();

  if (!CompilerInvocation::CreateFromArgs(*OriginalInvocation, CommandArgs,
                                          *DiagEngineWithCmdAndOpts->DiagEngine,
                                          Command.getExecutable())) {
    DiagEngineWithCmdAndOpts->DiagEngine->Report(
        diag::err_fe_expected_compiler_job)
        << llvm::join(CommandLine, " ");
    return false;
  }

  if (any(Worker.Service.getOptimizeArgs() & ScanningOptimizations::Macros))
    canonicalizeDefines(OriginalInvocation->getPreprocessorOpts());

  // Create the CompilerInstance.
  IntrusiveRefCntPtr<ModuleCache> ModCache =
      makeInProcessModuleCache(Worker.Service.getModuleCacheEntries());
  CIPtr = std::make_unique<CompilerInstance>(
      std::make_shared<CompilerInvocation>(*OriginalInvocation),
      Worker.PCHContainerOps, ModCache.get());
  auto &CI = *CIPtr;

  if (!initializeScanCompilerInstance(
          CI, OverlayFS, DiagEngineWithCmdAndOpts->DiagEngine->getClient(),
          Worker.Service, Worker.DepFS))
    return false;

  StableDirs = getInitialStableDirs(CI);
  auto MaybePrebuiltModulesASTMap =
      computePrebuiltModulesASTMap(CI, StableDirs);
  if (!MaybePrebuiltModulesASTMap)
    return false;

  PrebuiltModuleASTMap = std::move(*MaybePrebuiltModulesASTMap);
  OutputOpts = takeAndUpdateDependencyOutputOptionsFrom(CI);

  // We do not create the target in initializeScanCompilerInstance because
  // setting it here is unique for by-name lookups. We create the target only
  // once here, and the information is reused for all computeDependencies calls.
  // We do not need to call createTarget explicitly if we go through
  // CompilerInstance::ExecuteAction to perform scanning.
  CI.createTarget();

  return true;
}

bool CompilerInstanceWithContext::computeDependencies(
    StringRef ModuleName, DependencyConsumer &Consumer,
    DependencyActionController &Controller) {
  assert(CIPtr && "CIPtr must be initialized before calling this method");
  auto &CI = *CIPtr;

  // We create this cleanup object because computeDependencies may exit
  // early with errors.
  auto CleanUp = llvm::make_scope_exit([&]() {
    CI.clearDependencyCollectors();
    // The preprocessor may not be created at the entry of this method,
    // but it must have been created when this method returns, whether
    // there are errors during scanning or not.
    CI.getPreprocessor().removePPCallbacks();
  });

  auto MDC = initializeScanInstanceDependencyCollector(
      CI, std::make_unique<DependencyOutputOptions>(*OutputOpts), CWD, Consumer,
      Worker.Service,
      /* The MDC's constructor makes a copy of the OriginalInvocation, so
      we can pass it in without worrying that it might be changed across
      invocations of computeDependencies. */
      *OriginalInvocation, Controller, PrebuiltModuleASTMap, StableDirs);

  if (!SrcLocOffset) {
    // When SrcLocOffset is zero, we are at the beginning of the fake source
    // file. In this case, we call BeginSourceFile to initialize.
    std::unique_ptr<FrontendAction> Action =
        std::make_unique<PreprocessOnlyAction>();
    auto InputFile = CI.getFrontendOpts().Inputs.begin();
    bool ActionBeginSucceeded = Action->BeginSourceFile(CI, *InputFile);
    assert(ActionBeginSucceeded && "Action BeginSourceFile must succeed");
    (void)ActionBeginSucceeded;
  }

  Preprocessor &PP = CI.getPreprocessor();
  SourceManager &SM = PP.getSourceManager();
  FileID MainFileID = SM.getMainFileID();
  SourceLocation FileStart = SM.getLocForStartOfFile(MainFileID);
  SourceLocation IDLocation = FileStart.getLocWithOffset(SrcLocOffset);
  PPCallbacks *CB = nullptr;
  if (!SrcLocOffset) {
    // We need to call EnterSourceFile when SrcLocOffset is zero to initialize
    // the preprocessor.
    bool PPFailed = PP.EnterSourceFile(MainFileID, nullptr, SourceLocation());
    assert(!PPFailed && "Preprocess must be able to enter the main file.");
    (void)PPFailed;
    CB = MDC->getPPCallbacks();
  } else {
    // When SrcLocOffset is non-zero, the preprocessor has already been
    // initialized through a previous call of computeDependencies. We want to
    // preserve the PP's state, hence we do not call EnterSourceFile again.
    MDC->attachToPreprocessor(PP);
    CB = MDC->getPPCallbacks();

    FileID PrevFID;
    SrcMgr::CharacteristicKind FileType = SM.getFileCharacteristic(IDLocation);
    CB->LexedFileChanged(MainFileID,
                         PPChainedCallbacks::LexedFileChangeReason::EnterFile,
                         FileType, PrevFID, IDLocation);
  }

  SrcLocOffset++;
  SmallVector<IdentifierLoc, 2> Path;
  IdentifierInfo *ModuleID = PP.getIdentifierInfo(ModuleName);
  Path.emplace_back(IDLocation, ModuleID);
  auto ModResult = CI.loadModule(IDLocation, Path, Module::Hidden, false);

  assert(CB && "Must have PPCallbacks after module loading");
  CB->moduleImport(SourceLocation(), Path, ModResult);
  // Note that we are calling the CB's EndOfMainFile function, which
  // forwards the results to the dependency consumer.
  // It does not indicate the end of processing the fake file.
  CB->EndOfMainFile();

  if (!ModResult)
    return false;

  CompilerInvocation ModuleInvocation(*OriginalInvocation);
  MDC->applyDiscoveredDependencies(ModuleInvocation);
  Consumer.handleBuildCommand(
      {CommandLine[0], ModuleInvocation.getCC1CommandLine()});

  return true;
}

bool CompilerInstanceWithContext::finalize() {
  DiagConsumer->finish();
  return true;
}

llvm::Error CompilerInstanceWithContext::handleReturnStatus(bool Success) {
  assert(DiagPrinterWithOS && "Must use the default DiagnosticConsumer.");
  return Success ? llvm::Error::success()
                 : llvm::make_error<llvm::StringError>(
                       DiagPrinterWithOS->DiagnosticsOS.str(),
                       llvm::inconvertibleErrorCode());
}
