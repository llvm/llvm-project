//===- DependencyScanningWorker.cpp - Thread-Safe Scanning Worker ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/DependencyScanning/DependencyScanningWorker.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticFrontend.h"
#include "clang/Basic/DiagnosticSerialization.h"
#include "clang/DependencyScanning/DependencyActionController.h"
#include "clang/DependencyScanning/DependencyConsumer.h"
#include "clang/DependencyScanning/DependencyScanningFilesystem.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Sema/SemaOpenACC.h"
#include "clang/Serialization/ObjectFilePCHContainerReader.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/AdvisoryLock.h"
#include "llvm/Support/CrashRecoveryContext.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/TargetParser/Host.h"
#include <mutex>
#include <thread>

using namespace clang;
using namespace dependencies;

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
namespace {
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
                                 bool isOverridden, time_t StoredTime,
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
  void visitModuleFile(ModuleFileName Filename, serialization::ModuleKind Kind,
                       bool DirectlyImported) override {
    // If the CurrentFile is not
    // considered stable, update any of it's transitive dependents.
    auto PrebuiltEntryIt = PrebuiltModulesASTMap.find(CurrentFile);
    if ((PrebuiltEntryIt != PrebuiltModulesASTMap.end()) &&
        !PrebuiltEntryIt->second.isInStableDir())
      PrebuiltEntryIt->second.updateDependentsNotInStableDirs(
          PrebuiltModulesASTMap);
    CurrentFile = Filename.str();
  }

  /// Check the header search options for a given module when considering
  /// if the module comes from stable directories.
  bool ReadHeaderSearchOptions(const HeaderSearchOptions &HSOpts,
                               StringRef ModuleFilename, StringRef ContextHash,
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
} // namespace

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

  Listener.visitModuleFile(ModuleFileName::makeExplicit(PrebuiltModuleFilename),
                           serialization::MK_ExplicitModule,
                           /*DirectlyImported=*/true);
  if (ASTReader::readASTFileControlBlock(
          PrebuiltModuleFilename, CI.getFileManager(), CI.getModuleCache(),
          CI.getPCHContainerReader(),
          /*FindModuleFileExtensions=*/false, Listener,
          /*ValidateDiagnosticOptions=*/false, ASTReader::ARR_OutOfDate))
    return true;

  while (!Worklist.empty()) {
    // FIXME: This is assuming the PCH only refers to explicitly-built modules,
    // which technically is not guaranteed. To remove the assumption, we'd need
    // to also rework how the module files are handled to the scan, specifically
    // change the values of HeaderSearchOptions::PrebuiltModuleFiles from plain
    // paths to ModuleFileName.
    Listener.visitModuleFile(ModuleFileName::makeExplicit(Worklist.back()),
                             serialization::MK_ExplicitModule,
                             /*DirectlyImported=*/false);
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

namespace {
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
static void sanitizeDiagOpts(DiagnosticOptions &DiagOpts) {
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

static std::unique_ptr<CompilerInvocation>
createCompilerInvocation(ArrayRef<std::string> CommandLine,
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

static void initializeScanCompilerInstance(
    CompilerInstance &ScanInstance,
    IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS,
    DiagnosticConsumer *DiagConsumer, DependencyScanningService &Service,
    IntrusiveRefCntPtr<DependencyScanningWorkerFilesystem> DepFS) {
  ScanInstance.setBuildingModule(false);
  ScanInstance.createVirtualFileSystem(FS, DiagConsumer);
  ScanInstance.createDiagnostics(DiagConsumer, /*ShouldOwnClient=*/false);
  if (!Service.getOpts().EmitWarnings)
    ScanInstance.getDiagnostics().setIgnoreAllWarnings(true);
  ScanInstance.createFileManager();
  ScanInstance.createSourceManager();

  // Use DepFS for getting the dependency directives if requested to do so.
  if (Service.getOpts().Mode == ScanningMode::DependencyDirectivesScan)
    ScanInstance.setDependencyDirectivesGetter(
        std::make_unique<ScanningDependencyDirectivesGetter>(
            ScanInstance.getFileManager()));
}

static std::shared_ptr<CompilerInvocation>
createScanCompilerInvocation(const CompilerInvocation &Invocation,
                             const DependencyScanningService &Service,
                             DependencyActionController &Controller) {
  auto ScanInvocation = std::make_shared<CompilerInvocation>(Invocation);

  sanitizeDiagOpts(ScanInvocation->getDiagnosticOpts());

  ScanInvocation->getPreprocessorOpts().AllowPCHWithDifferentModulesCachePath =
      true;

  if (ScanInvocation->getHeaderSearchOpts().ModulesValidateOncePerBuildSession)
    ScanInvocation->getHeaderSearchOpts().BuildSessionTimestamp =
        Service.getOpts().BuildSessionTimestamp;

  ScanInvocation->getFrontendOpts().DisableFree = false;
  ScanInvocation->getFrontendOpts().GenerateGlobalModuleIndex = false;
  ScanInvocation->getFrontendOpts().UseGlobalModuleIndex = false;
  ScanInvocation->getFrontendOpts().GenReducedBMI = false;
  ScanInvocation->getFrontendOpts().ModuleOutputPath.clear();
  // This will prevent us compiling individual modules asynchronously since
  // FileManager is not thread-safe, but it does improve performance for now.
  ScanInvocation->getFrontendOpts().ModulesShareFileManager = true;
  ScanInvocation->getHeaderSearchOpts().ModuleFormat = "raw";
  ScanInvocation->getHeaderSearchOpts().ModulesIncludeVFSUsage =
      any(Service.getOpts().OptimizeArgs & ScanningOptimizations::VFS);

  // Consider different header search and diagnostic options to create
  // different modules. This avoids the unsound aliasing of module PCMs.
  //
  // TODO: Implement diagnostic bucketing to reduce the impact of strict
  // context hashing.
  ScanInvocation->getHeaderSearchOpts().ModulesStrictContextHash = true;
  ScanInvocation->getHeaderSearchOpts().ModulesSerializeOnlyPreprocessor = true;
  ScanInvocation->getHeaderSearchOpts().ModulesSkipDiagnosticOptions = true;
  ScanInvocation->getHeaderSearchOpts().ModulesSkipHeaderSearchPaths = true;
  ScanInvocation->getHeaderSearchOpts().ModulesSkipPragmaDiagnosticMappings =
      true;
  ScanInvocation->getHeaderSearchOpts().ModulesForceValidateUserHeaders = false;

  // FIXME: Do this even with PCHs by marking the option as something like
  // "preprocessor benign" in LangOptions.def so that it passes the
  // compatibility checks in ASTReader.
  if (ScanInvocation->getPreprocessorOpts().ImplicitPCHInclude.empty()) {
    // Application extension only affects the handling of availability
    // attributes, which cannot change the dependencies.
    ScanInvocation->getLangOpts().AppExt = false;
  }

  // Ensure that the scanner does not create new dependency collectors,
  // and thus won't write out the extra '.d' files to disk.
  ScanInvocation->getDependencyOutputOpts() = {};

  Controller.initializeScanInvocation(*ScanInvocation);

  return ScanInvocation;
}

static llvm::SmallVector<StringRef>
getInitialStableDirs(const CompilerInstance &ScanInstance) {
  // Create a collection of stable directories derived from the ScanInstance
  // for determining whether module dependencies would fully resolve from
  // those directories.
  llvm::SmallVector<StringRef> StableDirs;
  const StringRef Sysroot = ScanInstance.getHeaderSearchOpts().Sysroot;
  if (!Sysroot.empty() && (llvm::sys::path::root_directory(Sysroot) != Sysroot))
    StableDirs = {Sysroot, ScanInstance.getHeaderSearchOpts().ResourceDir};
  return StableDirs;
}

static std::optional<PrebuiltModulesAttrsMap>
computePrebuiltModulesASTMap(CompilerInstance &ScanInstance,
                             llvm::SmallVector<StringRef> &StableDirs) {
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

static std::unique_ptr<DependencyOutputOptions>
createDependencyOutputOptions(const CompilerInvocation &Invocation) {
  auto Opts = std::make_unique<DependencyOutputOptions>(
      Invocation.getDependencyOutputOpts());
  // We need at least one -MT equivalent for the generator of make dependency
  // files to work.
  if (Opts->Targets.empty())
    Opts->Targets = {deduceDepTarget(Invocation.getFrontendOpts().OutputFile,
                                     Invocation.getFrontendOpts().Inputs)};
  Opts->IncludeSystemHeaders = true;

  return Opts;
}

static std::shared_ptr<ModuleDepCollector>
initializeScanInstanceDependencyCollector(
    CompilerInstance &ScanInstance,
    std::unique_ptr<DependencyOutputOptions> DepOutputOpts,
    DependencyScanningService &Service, CompilerInvocation &Inv,
    DependencyActionController &Controller,
    PrebuiltModulesAttrsMap PrebuiltModulesASTMap,
    SmallVector<StringRef> &StableDirs) {
  auto MDC = std::make_shared<ModuleDepCollector>(
      Service, std::move(DepOutputOpts), ScanInstance, Controller, Inv,
      std::move(PrebuiltModulesASTMap), StableDirs);
  ScanInstance.addDependencyCollector(MDC);
  return MDC;
}

namespace {
/// Manages (and terminates) the asynchronous compilation of modules.
class AsyncModuleCompiles {
  std::mutex Mutex;
  bool Stop = false;
  // FIXME: Have the service own a thread pool and use that instead.
  std::vector<std::thread> Compiles;

public:
  /// Registers the module compilation, unless this instance is about to be
  /// destroyed.
  void add(llvm::unique_function<void()> Compile) {
    std::lock_guard<std::mutex> Lock(Mutex);
    if (!Stop)
      Compiles.emplace_back(std::move(Compile));
  }

  ~AsyncModuleCompiles() {
    {
      std::lock_guard<std::mutex> Lock(Mutex);
      Stop = true;
    }
    for (std::thread &Compile : Compiles)
      Compile.join();
  }
};

struct SingleModuleWithAsyncModuleCompiles : PreprocessOnlyAction {
  DependencyScanningService &Service;
  DependencyActionController &Controller;
  AsyncModuleCompiles &Compiles;

  SingleModuleWithAsyncModuleCompiles(DependencyScanningService &Service,
                                      DependencyActionController &Controller,
                                      AsyncModuleCompiles &Compiles)
      : Service(Service), Controller(Controller), Compiles(Compiles) {}

  bool BeginSourceFileAction(CompilerInstance &CI) override;
};

/// Runs the preprocessor on a TU with single-module-parse-mode and compiles
/// modules asynchronously without blocking or importing them.
struct SingleTUWithAsyncModuleCompiles : PreprocessOnlyAction {
  DependencyScanningService &Service;
  DependencyActionController &Controller;
  AsyncModuleCompiles &Compiles;

  SingleTUWithAsyncModuleCompiles(DependencyScanningService &Service,
                                  DependencyActionController &Controller,
                                  AsyncModuleCompiles &Compiles)
      : Service(Service), Controller(Controller), Compiles(Compiles) {}

  bool BeginSourceFileAction(CompilerInstance &CI) override;
};

/// The preprocessor callback that takes care of initiating an asynchronous
/// module compilation if needed.
struct AsyncModuleCompile : PPCallbacks {
  CompilerInstance &CI;
  DependencyScanningService &Service;
  DependencyActionController &Controller;
  AsyncModuleCompiles &Compiles;

  AsyncModuleCompile(CompilerInstance &CI, DependencyScanningService &Service,
                     DependencyActionController &Controller,
                     AsyncModuleCompiles &Compiles)
      : CI(CI), Service(Service), Controller(Controller), Compiles(Compiles) {}

  void moduleLoadSkipped(Module *M) override {
    M = M->getTopLevelModule();

    HeaderSearch &HS = CI.getPreprocessor().getHeaderSearchInfo();
    ModuleCache &ModCache = CI.getModuleCache();
    ModuleFileName ModuleFileName = HS.getCachedModuleFileName(M);

    uint64_t Timestamp = ModCache.getModuleTimestamp(ModuleFileName);
    // Someone else already built/validated the PCM.
    if (Timestamp > CI.getHeaderSearchOpts().BuildSessionTimestamp)
      return;

    if (!CI.getASTReader())
      CI.createASTReader();
    SmallVector<ASTReader::ImportedModule, 0> Imported;
    // Only calling ReadASTCore() to avoid the expensive eager deserialization
    // of the clang::Module objects in ReadAST().
    // FIXME: Consider doing this in the new thread depending on how expensive
    // the read turns out to be.
    switch (CI.getASTReader()->ReadASTCore(
        ModuleFileName, serialization::MK_ImplicitModule, SourceLocation(),
        nullptr, Imported, {}, {}, {},
        ASTReader::ARR_OutOfDate | ASTReader::ARR_Missing |
            ASTReader::ARR_TreatModuleWithErrorsAsOutOfDate)) {
    case ASTReader::Success:
      // We successfully read a valid, up-to-date PCM.
      // FIXME: This could update the timestamp. Regular calls to
      // ASTReader::ReadAST() would do so unless they encountered corrupted
      // AST block, corrupted extension block, or did not read the expected
      // top-level module.
      return;
    case ASTReader::OutOfDate:
    case ASTReader::Missing:
      // The most interesting case.
      break;
    default:
      // Let the regular scan diagnose this.
      return;
    }

    auto Lock = ModCache.getLock(ModuleFileName);
    bool Owned;
    llvm::Error LockErr = Lock->tryLock().moveInto(Owned);
    // Someone else is building the PCM right now.
    if (!LockErr && !Owned)
      return;
    // We should build the PCM.
    IntrusiveRefCntPtr<llvm::vfs::FileSystem> VFS =
        llvm::makeIntrusiveRefCnt<DependencyScanningWorkerFilesystem>(
            Service, Service.getOpts().MakeVFS());
    VFS = createVFSFromCompilerInvocation(CI.getInvocation(),
                                          CI.getDiagnostics(), std::move(VFS));
    auto DC = std::make_unique<DiagnosticConsumer>();
    auto MC = makeInProcessModuleCache(Service.getModuleCacheEntries(),
                                       Service.getLogger());
    CompilerInstance::ThreadSafeCloneConfig CloneConfig(std::move(VFS), *DC,
                                                        std::move(MC));
    auto ModCI1 = CI.cloneForModuleCompile(SourceLocation(), M, ModuleFileName,
                                           CloneConfig);
    auto ModCI2 = CI.cloneForModuleCompile(SourceLocation(), M, ModuleFileName,
                                           CloneConfig);

    auto ModController = Controller.clone();

    // Note: This lock belongs to a module cache that might not outlive the
    // thread. This works, because the in-process lock only refers to an object
    // managed by the service, which does outlive the thread.
    Compiles.add([Lock = std::move(Lock), ModCI1 = std::move(ModCI1),
                  ModCI2 = std::move(ModCI2), DC = std::move(DC),
                  ModController = std::move(ModController), Service = &Service,
                  Compiles = &Compiles] {
      llvm::CrashRecoveryContext CRC;
      (void)CRC.RunSafely([&] {
        // Quickly discovers and compiles modules for the real scan below.
        SingleModuleWithAsyncModuleCompiles Action1(*Service, *ModController,
                                                    *Compiles);
        (void)ModCI1->ExecuteAction(Action1);
        // The real scan below.
        ModCI2->getPreprocessorOpts().SingleModuleParseMode = false;
        GenerateModuleFromModuleMapAction Action2;
        (void)ModCI2->ExecuteAction(Action2);
      });
    });
  }
};

bool SingleModuleWithAsyncModuleCompiles::BeginSourceFileAction(
    CompilerInstance &CI) {
  CI.getInvocation().getPreprocessorOpts().SingleModuleParseMode = true;
  CI.getPreprocessor().addPPCallbacks(
      std::make_unique<AsyncModuleCompile>(CI, Service, Controller, Compiles));
  return true;
}

bool SingleTUWithAsyncModuleCompiles::BeginSourceFileAction(
    CompilerInstance &CI) {
  CI.getInvocation().getPreprocessorOpts().SingleModuleParseMode = true;
  CI.getPreprocessor().addPPCallbacks(
      std::make_unique<AsyncModuleCompile>(CI, Service, Controller, Compiles));
  return true;
}
} // namespace

static void runTUModulePrescan(CompilerInstance &PrescanCI,
                               DependencyScanningService &Service,
                               DependencyActionController &Controller,
                               AsyncModuleCompiles &Compiles) {
  SingleTUWithAsyncModuleCompiles Action(Service, Controller, Compiles);
  (void)PrescanCI.ExecuteAction(Action);
}

namespace clang {
namespace dependencies {

std::unique_ptr<DiagnosticOptions>
createScanningDiagOptions(ArrayRef<std::string> CommandLine) {
  std::vector<const char *> CCommandLine(CommandLine.size(), nullptr);
  llvm::transform(CommandLine, CCommandLine.begin(),
                  [](const std::string &Str) { return Str.c_str(); });
  auto DiagOpts = CreateAndPopulateDiagOpts(CCommandLine);
  sanitizeDiagOpts(*DiagOpts);
  return DiagOpts;
}

class CompilerInstanceWithContext {
  // Context
  DependencyScanningWorker &Worker;
  llvm::StringRef CWD;
  std::vector<std::string> CommandLine;

  // Context - compiler invocation
  std::unique_ptr<CompilerInvocation> OriginalInvocation;

  // Context - output options
  std::unique_ptr<DependencyOutputOptions> OutputOpts;

  // Context - stable directory handling
  llvm::SmallVector<StringRef> StableDirs;
  PrebuiltModulesAttrsMap PrebuiltModuleASTMap;

  // Context - used by AsyncScan's prescan pass
  IntrusiveRefCntPtr<llvm::vfs::FileSystem> ScanFS;

  // Compiler Instance
  std::unique_ptr<CompilerInstance> CIPtr;

  // Source location offset.
  int32_t SrcLocOffset = 0;

  CompilerInstanceWithContext(DependencyScanningWorker &Worker, StringRef CWD,
                              ArrayRef<std::string> CMD)
      : Worker(Worker), CWD(CWD), CommandLine(CMD.begin(), CMD.end()) {}

  bool initialize(DependencyActionController &Controller,
                  DiagnosticsEngine &DiagEngine,
                  IntrusiveRefCntPtr<llvm::vfs::FileSystem> OverlayFS) {
    {
      auto LogLine = Worker.Service.getLogger().log();
      LogLine.logArray("init_compiler_instance_with_context:", " ",
                       CommandLine);
    }

    ScanFS = Worker.makeEffectiveVFS(CWD, std::move(OverlayFS));
    OriginalInvocation = createCompilerInvocation(CommandLine, DiagEngine);
    if (!OriginalInvocation) {
      DiagEngine.Report(diag::err_fe_expected_compiler_job)
          << llvm::join(CommandLine, " ");
      return false;
    }

    return initializeScanInstance(Controller, DiagEngine.getClient());
  }

  bool initializeScanInstance(DependencyActionController &Controller,
                              DiagnosticConsumer *DiagConsumer) {
    assert(OriginalInvocation && ScanFS &&
           "OriginalInvocation and ScanFS must be set before this call");

    if (any(Worker.Service.getOpts().OptimizeArgs &
            ScanningOptimizations::Macros))
      canonicalizeDefines(OriginalInvocation->getPreprocessorOpts());

    // Create the CompilerInstance.
    std::shared_ptr<ModuleCache> ModCache = makeInProcessModuleCache(
        Worker.Service.getModuleCacheEntries(), Worker.Service.getLogger());
    CIPtr = std::make_unique<CompilerInstance>(
        createScanCompilerInvocation(*OriginalInvocation, Worker.Service,
                                     Controller),
        Worker.PCHContainerOps, std::move(ModCache));
    auto &CI = *CIPtr;

    initializeScanCompilerInstance(CI, ScanFS, DiagConsumer, Worker.Service,
                                   Worker.DepFS);

    StableDirs = getInitialStableDirs(CI);
    auto MaybePrebuiltModulesASTMap =
        computePrebuiltModulesASTMap(CI, StableDirs);
    if (!MaybePrebuiltModulesASTMap)
      return false;

    PrebuiltModuleASTMap = std::move(*MaybePrebuiltModulesASTMap);
    OutputOpts = createDependencyOutputOptions(*OriginalInvocation);

    // We do not create the target in initializeScanCompilerInstance because
    // setting it here is unique for by-name lookups. We create the target only
    // once here, and the information is reused for all computeDependencies
    // calls. We do not need to call createTarget explicitly if we go through
    // CompilerInstance::ExecuteAction to perform scanning.
    return CI.createTarget();
  }

  bool prescanModulesAsync(AsyncModuleCompiles &Compiles,
                           DependencyActionController &Controller) {
    auto ModCache = makeInProcessModuleCache(
        Worker.Service.getModuleCacheEntries(), Worker.Service.getLogger());
    CompilerInstance PrescanCI(
        std::make_shared<CompilerInvocation>(CIPtr->getInvocation()),
        Worker.PCHContainerOps, std::move(ModCache));

    DiagnosticConsumer DiagConsumer;
    initializeScanCompilerInstance(PrescanCI, ScanFS, &DiagConsumer,
                                   Worker.Service, Worker.DepFS);

    // FIXME: reuse the StableDirs/PrebuiltModuleASTMap computed in
    // initialize().
    SmallVector<StringRef> PrescanStableDirs = getInitialStableDirs(PrescanCI);
    if (!computePrebuiltModulesASTMap(PrescanCI, PrescanStableDirs))
      return false;

    if (PrescanCI.getFrontendOpts().ProgramAction == frontend::GeneratePCH)
      PrescanCI.getLangOpts().CompilingPCH = true;

    runTUModulePrescan(PrescanCI, Worker.Service, Controller, Compiles);
    return true;
  }

public:
  static std::optional<CompilerInstanceWithContext>
  initializeFromCC1Commandline(
      DependencyScanningWorker &Worker, StringRef CWD,
      ArrayRef<std::string> CC1CommandLine, DiagnosticsEngine &DiagEngine,
      IntrusiveRefCntPtr<llvm::vfs::FileSystem> OverlayFS,
      DependencyActionController &Controller) {
    CompilerInstanceWithContext CIWC(Worker, CWD, CC1CommandLine);
    if (!CIWC.initialize(Controller, DiagEngine, std::move(OverlayFS)))
      return std::nullopt;
    return std::move(CIWC);
  }

  bool computeDependencies(StringRef ModuleName, DependencyConsumer &Consumer,
                           DependencyActionController &Controller) {
    Worker.Service.getLogger().log() << "start scan_by_name: " << ModuleName;
    llvm::scope_exit ExitLogging([&] {
      Worker.Service.getLogger().log() << "finish scan_by_name: " << ModuleName;
    });
    if (SrcLocOffset >= DependencyScanningWorker::MaxNumOfByNameQueries)
      llvm::report_fatal_error("exceeded maximum by-name scans for worker");

    assert(CIPtr && "CIPtr must be initialized before calling this method");
    auto &CI = *CIPtr;

    // We need to reset the diagnostics, so that the diagnostics issued
    // during a previous computeDependencies call do not affect the current
    // call. If we do not reset, we may inherit fatal errors from a previous
    // call.
    CI.getDiagnostics().Reset();

    // We create this cleanup object because computeDependencies may exit
    // early with errors.
    llvm::scope_exit CleanUp([&]() {
      CI.clearDependencyCollectors();
      // The preprocessor may not be created at the entry of this method,
      // but it must have been created when this method returns, whether
      // there are errors during scanning or not.
      CI.getPreprocessor().removePPCallbacks();
    });

    auto MDC = initializeScanInstanceDependencyCollector(
        CI, std::make_unique<DependencyOutputOptions>(*OutputOpts),
        Worker.Service,
        /* The MDC's constructor makes a copy of the OriginalInvocation, so
        we can pass it in without worrying that it might be changed across
        invocations of computeDependencies. */
        *OriginalInvocation, Controller, PrebuiltModuleASTMap, StableDirs);

    CompilerInvocation ModuleInvocation(*OriginalInvocation);
    if (!Controller.initialize(CI, ModuleInvocation))
      return false;

    if (!SrcLocOffset) {
      // When SrcLocOffset is zero, we are at the beginning of the fake source
      // file. In this case, we call BeginSourceFile to initialize.
      std::unique_ptr<FrontendAction> Action =
          std::make_unique<PreprocessOnlyAction>();
      auto *InputFile = CI.getFrontendOpts().Inputs.begin();
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
      SrcMgr::CharacteristicKind FileType =
          SM.getFileCharacteristic(IDLocation);
      CB->LexedFileChanged(MainFileID,
                           PPChainedCallbacks::LexedFileChangeReason::EnterFile,
                           FileType, PrevFID, IDLocation);
    }

    // FIXME: Scan modules asynchronously here as well.

    SrcLocOffset++;
    SmallVector<IdentifierLoc, 2> Path;
    IdentifierInfo *ModuleID = PP.getIdentifierInfo(ModuleName);
    Path.emplace_back(IDLocation, ModuleID);
    auto ModResult = CI.loadModule(IDLocation, Path, Module::Hidden, false);

    assert(CB && "Must have PPCallbacks after module loading");
    CB->moduleImport(SourceLocation(), Path, ModResult);

    if (!ModResult)
      return false;

    if (CI.getDiagnostics().hasErrorOccurred())
      return false;

    MDC->run(Consumer);
    MDC->applyDiscoveredDependencies(ModuleInvocation);

    bool Success = ModuleInvocation.withCowRef<bool>(
        [&](CowCompilerInvocation &CowModuleInvocation) {
          return Controller.finalize(CI, CowModuleInvocation);
        });
    if (!Success)
      return false;

    Consumer.handleBuildCommand(
        {CommandLine[0], ModuleInvocation.getCC1CommandLine()});

    return true;
  }

  std::shared_ptr<ModuleDepCollector>
  scanTranslationUnit(DependencyConsumer &Consumer,
                      DependencyActionController &Controller) {
    assert(CIPtr && "CIPtr must be initialized before calling this method");
    auto &CI = *CIPtr;

    std::optional<AsyncModuleCompiles> AsyncCompiles;
    if (Worker.Service.getOpts().AsyncScanModules) {
      AsyncCompiles.emplace();
      if (!prescanModulesAsync(*AsyncCompiles, Controller))
        return nullptr;
    }

    auto MDC = initializeScanInstanceDependencyCollector(
        CI, std::make_unique<DependencyOutputOptions>(*OutputOpts),
        Worker.Service, *OriginalInvocation, Controller, PrebuiltModuleASTMap,
        StableDirs);

    if (CI.getDiagnostics().hasErrorOccurred())
      return nullptr;

    if (!Controller.initialize(CI, *OriginalInvocation))
      return nullptr;

    ReadPCHAndPreprocessAction Action;
    if (!CI.ExecuteAction(Action))
      return nullptr;

    MDC->run(Consumer);
    if (!applyAndReport(*MDC, *OriginalInvocation, Consumer, Controller,
                        CommandLine[0]))
      return nullptr;
    return MDC;
  }

  bool applyAndReport(ModuleDepCollector &MDC,
                      CompilerInvocation &ModuleInvocation,
                      DependencyConsumer &Consumer,
                      DependencyActionController &Controller,
                      StringRef Executable) {
    MDC.applyDiscoveredDependencies(ModuleInvocation);
    bool Success = ModuleInvocation.withCowRef<bool>(
        [&](CowCompilerInvocation &CowModuleInvocation) {
          return Controller.finalize(*CIPtr, CowModuleInvocation);
        });
    if (!Success)
      return false;
    Consumer.handleBuildCommand(
        {Executable.str(), ModuleInvocation.getCC1CommandLine()});
    return true;
  }
};
} // namespace dependencies
} // namespace clang

DependencyScanningWorker::DependencyScanningWorker(
    DependencyScanningService &Service)
    : Service(Service) {
  PCHContainerOps = std::make_shared<PCHContainerOperations>();
  // We need to read object files from PCH built outside the scanner.
  PCHContainerOps->registerReader(
      std::make_unique<ObjectFilePCHContainerReader>());
  // The scanner itself writes only raw ast files.
  PCHContainerOps->registerWriter(std::make_unique<RawPCHContainerWriter>());

  auto BaseFS = Service.getOpts().MakeVFS();

  if (Service.getOpts().TraceVFS) {
    TracingFS = llvm::makeIntrusiveRefCnt<llvm::vfs::TracingFileSystem>(
        std::move(BaseFS));
    BaseFS = TracingFS;
  }

  DepFS = llvm::makeIntrusiveRefCnt<DependencyScanningWorkerFilesystem>(
      Service, std::move(BaseFS));
}

DependencyScanningWorker::~DependencyScanningWorker() = default;

IntrusiveRefCntPtr<llvm::vfs::FileSystem>
DependencyScanningWorker::makeEffectiveVFS(
    StringRef WorkingDirectory,
    IntrusiveRefCntPtr<llvm::vfs::FileSystem> OverlayFS) const {
  IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS = DepFS;
  if (OverlayFS) {
    auto NewFS =
        llvm::makeIntrusiveRefCnt<llvm::vfs::OverlayFileSystem>(std::move(FS));
    NewFS->pushOverlay(std::move(OverlayFS));
    FS = std::move(NewFS);
  }
  FS->setCurrentWorkingDirectory(WorkingDirectory);
  return FS;
}

bool DependencyScanningWorker::computeDependencies(
    StringRef WorkingDirectory, ArrayRef<ArrayRef<std::string>> CommandLines,
    DependencyConsumer &DepConsumer, DependencyActionController &Controller,
    DiagnosticConsumer &DiagConsumer,
    IntrusiveRefCntPtr<llvm::vfs::FileSystem> OverlayFS) {
  auto FS = makeEffectiveVFS(WorkingDirectory, OverlayFS);

  bool Scanned = false;
  std::shared_ptr<ModuleDepCollector> MDC;
  std::optional<CompilerInstanceWithContext> CIWC;

  const bool Success = llvm::all_of(CommandLines, [&](const auto &Cmd) {
    if (StringRef(Cmd[1]) != "-cc1") {
      // Non-clang command. Just pass through to the dependency consumer.
      DepConsumer.handleBuildCommand(
          {Cmd.front(), {Cmd.begin() + 1, Cmd.end()}});
      return true;
    }

    Service.getLogger().log().logArray("starting scanning command:", " ", Cmd);
    llvm::scope_exit ExitLogging([&] {
      Service.getLogger().log().logArray("finished scanning command:", " ",
                                         Cmd);
    });

    auto DiagOpts = createScanningDiagOptions(Cmd);
    auto DiagEngine =
        CompilerInstance::createDiagnostics(*FS, *DiagOpts, &DiagConsumer,
                                            /*ShouldOwnClient=*/false);
    if (!Scanned) {
      // Scanning runs once for the first -cc1 invocation in a chain of driver
      // jobs.
      // For any dependent jobs, reuse the scanning result and just update the
      // new invocation.
      // FIXME: to support multi-arch builds, each arch requires a separate
      // scan.
      Scanned = true;
      auto Result = CompilerInstanceWithContext::initializeFromCC1Commandline(
          *this, WorkingDirectory, Cmd, *DiagEngine, OverlayFS, Controller);
      if (!Result)
        return false;
      CIWC.emplace(std::move(*Result));
      MDC = CIWC->scanTranslationUnit(DepConsumer, Controller);
      return MDC != nullptr;
    }

    auto Invocation = createCompilerInvocation(Cmd, *DiagEngine);
    if (!Invocation)
      return false;

    // The first cc1 is canonicalized in initializeScanInstance; each sibling
    // invocation must likewise be canonicalized before its cc1 command line is
    // emitted. This is mostly relevant for multi-arch jobs where we currently
    // do not do re-scans.
    if (any(Service.getOpts().OptimizeArgs & ScanningOptimizations::Macros))
      canonicalizeDefines(Invocation->getPreprocessorOpts());

    assert(CIWC && "Must have an initialized CIWC");
    return CIWC->applyAndReport(*MDC, *Invocation, DepConsumer, Controller,
                                Cmd.front());
  });

  return Success && Scanned;
}

bool DependencyScanningWorker::computeDependenciesByName(
    StringRef CWD, ArrayRef<std::string> CC1CommandLine,
    IntrusiveRefCntPtr<llvm::vfs::FileSystem> OverlayFS,
    DiagnosticConsumer &DiagConsumer, DependencyActionController &Controller,
    llvm::function_ref<std::optional<std::string>()> getNextName,
    DependencyConsumer &DepConsumer) {
  auto FS = makeEffectiveVFS(CWD, OverlayFS);
  auto DiagOpts = createScanningDiagOptions(CC1CommandLine);
  auto DiagEngine =
      CompilerInstance::createDiagnostics(*FS, *DiagOpts, &DiagConsumer,
                                          /*ShouldOwnClient=*/false);
  std::optional<CompilerInstanceWithContext> CIWC =
      CompilerInstanceWithContext::initializeFromCC1Commandline(
          *this, CWD, CC1CommandLine, *DiagEngine, std::move(OverlayFS),
          Controller);
  if (!CIWC)
    return false;

  bool AllScansSucceeded = true;
  while (std::optional<std::string> NextName = getNextName()) {
    bool Success =
        CIWC->computeDependencies(*NextName, DepConsumer, Controller);
    DepConsumer.finishQuery(*NextName, Success);
    AllScansSucceeded = AllScansSucceeded && Success;
  }
  return AllScansSucceeded;
}
