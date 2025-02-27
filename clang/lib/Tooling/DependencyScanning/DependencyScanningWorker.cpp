//===- DependencyScanningWorker.cpp - clang-scan-deps worker --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/DependencyScanning/DependencyScanningWorker.h"
#include "clang/Basic/DiagnosticCAS.h"
#include "clang/Basic/DiagnosticDriver.h"
#include "clang/Basic/DiagnosticFrontend.h"
#include "clang/Basic/DiagnosticSerialization.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Job.h"
#include "clang/Driver/Tool.h"
#include "clang/Frontend/CompileJobCacheKey.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/MultiplexConsumer.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Frontend/Utils.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "clang/Serialization/ObjectFilePCHContainerReader.h"
#include "clang/Tooling/DependencyScanning/DependencyScanningService.h"
#include "clang/Tooling/DependencyScanning/ModuleDepCollector.h"
#include "clang/Tooling/DependencyScanning/ScanAndUpdateArgs.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/CAS/CASProvidingFileSystem.h"
#include "llvm/CAS/CachingOnDiskFileSystem.h"
#include "llvm/CAS/ObjectStore.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/PrefixMapper.h"
#include "llvm/TargetParser/Host.h"
#include <optional>

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

/// A listener that collects the imported modules and optionally the input
/// files.
class PrebuiltModuleListener : public ASTReaderListener {
public:
  PrebuiltModuleListener(CompilerInstance &CI,
                         PrebuiltModuleFilesT &PrebuiltModuleFiles,
                         llvm::SmallVector<std::string> &NewModuleFiles,
                         PrebuiltModuleVFSMapT &PrebuiltModuleVFSMap,
                         DiagnosticsEngine &Diags)
      : CI(CI), PrebuiltModuleFiles(PrebuiltModuleFiles),
        NewModuleFiles(NewModuleFiles),
        PrebuiltModuleVFSMap(PrebuiltModuleVFSMap), Diags(Diags) {}

  bool needsImportVisitation() const override { return true; }

  void visitImport(StringRef ModuleName, StringRef Filename) override {
    if (PrebuiltModuleFiles.insert({ModuleName.str(), Filename.str()}).second)
      NewModuleFiles.push_back(Filename.str());
  }

  void visitModuleFile(StringRef Filename,
                       serialization::ModuleKind Kind) override {
    CurrentFile = Filename;
  }

  bool ReadHeaderSearchPaths(const HeaderSearchOptions &HSOpts,
                             bool Complain) override {
    std::vector<std::string> VFSOverlayFiles = HSOpts.VFSOverlayFiles;
    PrebuiltModuleVFSMap.insert(
        {CurrentFile, llvm::StringSet<>(VFSOverlayFiles)});
    return checkHeaderSearchPaths(
        HSOpts, CI.getHeaderSearchOpts(), Complain ? &Diags : nullptr, CI.getLangOpts());
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
  llvm::SmallVector<std::string> &NewModuleFiles;
  PrebuiltModuleVFSMapT &PrebuiltModuleVFSMap;
  DiagnosticsEngine &Diags;
  std::string CurrentFile;
};

/// Visit the given prebuilt module and collect all of the modules it
/// transitively imports and contributing input files.
static bool visitPrebuiltModule(StringRef PrebuiltModuleFilename,
                                CompilerInstance &CI,
                                PrebuiltModuleFilesT &ModuleFiles,
                                PrebuiltModuleVFSMapT &PrebuiltModuleVFSMap,
                                DiagnosticsEngine &Diags) {
  // List of module files to be processed.
  llvm::SmallVector<std::string> Worklist;
  PrebuiltModuleListener Listener(CI, ModuleFiles, Worklist,
                                  PrebuiltModuleVFSMap, Diags);

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
        .Cases("pch-vfs-diff", "error=pch-vfs-diff", false)
        .StartsWith("no-error=", false)
        .Default(true);
  });
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

/// Builds a dependency file after reversing prefix mappings. This allows
/// emitting a .d file that has real paths where they would otherwise be
/// canonicalized.
class ReversePrefixMappingDependencyFileGenerator
    : public DependencyFileGenerator {
  llvm::PrefixMapper ReverseMapper;

public:
  ReversePrefixMappingDependencyFileGenerator(
      const DependencyOutputOptions &Opts)
      : DependencyFileGenerator(Opts) {}

  void initialize(const CompilerInvocation &CI) {
    llvm::PrefixMapper Mapper;
    DepscanPrefixMapping::configurePrefixMapper(CI, Mapper);
    if (Mapper.empty())
      return;

    ReverseMapper.addInverseRange(Mapper.getMappings());
    ReverseMapper.sort();
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
};

/// See \c WrapScanModuleBuildAction.
class WrapScanModuleBuildConsumer : public ASTConsumer {
public:
  WrapScanModuleBuildConsumer(CompilerInstance &CI,
                              DependencyActionController &Controller)
      : CI(CI), Controller(Controller) {}

  void HandleTranslationUnit(ASTContext &Ctx) override {
    if (auto E = Controller.finalizeModuleBuild(CI))
      Ctx.getDiagnostics().Report(diag::err_cas_depscan_failed) << std::move(E);
  }

private:
  CompilerInstance &CI;
  DependencyActionController &Controller;
};

/// A wrapper for implicit module build actions in the scanner.
class WrapScanModuleBuildAction : public WrapperFrontendAction {
public:
  WrapScanModuleBuildAction(std::unique_ptr<FrontendAction> WrappedAction,
                            DependencyActionController &Controller)
      : WrapperFrontendAction(std::move(WrappedAction)),
        Controller(Controller) {}

private:
  bool BeginInvocation(CompilerInstance &CI) override {
    if (auto E = Controller.initializeModuleBuild(CI)) {
      CI.getDiagnostics().Report(diag::err_cas_depscan_failed) << std::move(E);
      return false;
    }
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
    auto Consumer =
        std::make_unique<WrapScanModuleBuildConsumer>(CI, Controller);
    std::vector<std::unique_ptr<ASTConsumer>> Consumers;
    Consumers.push_back(std::move(Consumer));
    Consumers.push_back(std::move(OtherConsumer));
    return std::make_unique<MultiplexConsumer>(std::move(Consumers));
  }

private:
  DependencyActionController &Controller;
};

/// A clang tool that runs the preprocessor in a mode that's optimized for
/// dependency scanning for the given compiler invocation.
class DependencyScanningAction : public tooling::ToolAction {
public:
  DependencyScanningAction(
      DependencyScanningService &Service, StringRef WorkingDirectory,
      DependencyConsumer &Consumer, DependencyActionController &Controller,
      llvm::IntrusiveRefCntPtr<DependencyScanningWorkerFilesystem> DepFS,
      llvm::IntrusiveRefCntPtr<DependencyScanningCASFilesystem> DepCASFS,
      llvm::IntrusiveRefCntPtr<llvm::cas::CachingOnDiskFileSystem> CacheFS,
      bool DisableFree, bool EmitDependencyFile,
      bool DiagGenerationAsCompilation, const CASOptions &CASOpts,
      std::optional<StringRef> ModuleName = std::nullopt,
      raw_ostream *VerboseOS = nullptr)
      : Service(Service), WorkingDirectory(WorkingDirectory), Consumer(Consumer),
        Controller(Controller), DepFS(std::move(DepFS)),
        DepCASFS(std::move(DepCASFS)), CacheFS(std::move(CacheFS)),
        DisableFree(DisableFree),
        CASOpts(CASOpts), EmitDependencyFile(EmitDependencyFile),
        DiagGenerationAsCompilation(DiagGenerationAsCompilation),
        ModuleName(ModuleName), VerboseOS(VerboseOS) {}

  bool runInvocation(std::shared_ptr<CompilerInvocation> Invocation,
                     FileManager *DriverFileMgr,
                     std::shared_ptr<PCHContainerOperations> PCHContainerOps,
                     DiagnosticConsumer *DiagConsumer) override {
    // Make a deep copy of the original Clang invocation.
    CompilerInvocation OriginalInvocation(*Invocation);
    // Restore the value of DisableFree, which may be modified by Tooling.
    OriginalInvocation.getFrontendOpts().DisableFree = DisableFree;
    if (any(Service.getOptimizeArgs() & ScanningOptimizations::Macros))
      canonicalizeDefines(OriginalInvocation.getPreprocessorOpts());

    if (Scanned) {
      CompilerInstance &ScanInstance = *ScanInstanceStorage;
      auto reportError = [&ScanInstance](Error &&E) -> bool {
        ScanInstance.getDiagnostics().Report(diag::err_cas_depscan_failed)
            << std::move(E);
        return false;
      };

      // Scanning runs once for the first -cc1 invocation in a chain of driver
      // jobs. For any dependent jobs, reuse the scanning result and just
      // update the LastCC1Arguments to correspond to the new invocation.
      // FIXME: to support multi-arch builds, each arch requires a separate scan
      if (MDC)
        MDC->applyDiscoveredDependencies(OriginalInvocation);

      if (Error E = Controller.finalize(ScanInstance, OriginalInvocation))
        return reportError(std::move(E));

      LastCC1Arguments = OriginalInvocation.getCC1CommandLine();
      LastCC1CacheKey = Controller.getCacheKey(OriginalInvocation);
      return true;
    }

    Scanned = true;

    // Create a compiler instance to handle the actual work.
    ScanInstanceStorage.emplace(std::move(PCHContainerOps));
    CompilerInstance &ScanInstance = *ScanInstanceStorage;
    ScanInstance.setInvocation(std::move(Invocation));
    ScanInstance.getInvocation().getCASOpts() = CASOpts;

    // Create the compiler's actual diagnostics engine.
    if (!DiagGenerationAsCompilation)
      sanitizeDiagOpts(ScanInstance.getDiagnosticOpts());
    assert(!DiagConsumerFinished && "attempt to reuse finished consumer");
    ScanInstance.createDiagnostics(DriverFileMgr->getVirtualFileSystem(),
                                   DiagConsumer, /*ShouldOwnClient=*/false);
    if (!ScanInstance.hasDiagnostics())
      return false;
    if (VerboseOS)
      ScanInstance.setVerboseOutputStream(*VerboseOS);

    ScanInstance.getPreprocessorOpts().AllowPCHWithDifferentModulesCachePath =
        true;

    ScanInstance.getFrontendOpts().GenerateGlobalModuleIndex = false;
    ScanInstance.getFrontendOpts().UseGlobalModuleIndex = false;
    // This will prevent us compiling individual modules asynchronously since
    // FileManager is not thread-safe, but it does improve performance for now.
    ScanInstance.getFrontendOpts().ModulesShareFileManager = true;
    if (DepCASFS)
      ScanInstance.getFrontendOpts().ModulesShareFileManager = false;
    ScanInstance.getHeaderSearchOpts().ModuleFormat = "raw";
    ScanInstance.getHeaderSearchOpts().ModulesIncludeVFSUsage =
        any(Service.getOptimizeArgs() & ScanningOptimizations::VFS);

    // Support for virtual file system overlays.
    auto FS = createVFSFromCompilerInvocation(
        ScanInstance.getInvocation(), ScanInstance.getDiagnostics(),
        DriverFileMgr->getVirtualFileSystemPtr());

    // Use the dependency scanning optimized file system if requested to do so.
    if (DepFS) {
      StringRef ModulesCachePath =
          ScanInstance.getHeaderSearchOpts().ModuleCachePath;

      DepFS->resetBypassedPathPrefix();
      if (!ModulesCachePath.empty())
        DepFS->setBypassedPathPrefix(ModulesCachePath);

      ScanInstance.getPreprocessorOpts().DependencyDirectivesForFile =
          [LocalDepFS = DepFS](FileEntryRef File)
          -> std::optional<ArrayRef<dependency_directives_scan::Directive>> {
        if (llvm::ErrorOr<EntryRef> Entry =
                LocalDepFS->getOrCreateFileSystemEntry(File.getName()))
          if (LocalDepFS->ensureDirectiveTokensArePopulated(*Entry))
            return Entry->getDirectiveTokens();
        return std::nullopt;
      };
    }

    // CAS Implementation.
    if (DepCASFS)
      ScanInstance.getPreprocessorOpts().DependencyDirectivesForFile =
          [LocalDepCASFS = DepCASFS](FileEntryRef File) {
            return LocalDepCASFS->getDirectiveTokens(File.getName());
          };

    // Create a new FileManager to match the invocation's FileSystemOptions.
    auto *FileMgr = ScanInstance.createFileManager(FS);
    ScanInstance.createSourceManager(*FileMgr);

    // Store the list of prebuilt module files into header search options. This
    // will prevent the implicit build to create duplicate modules and will
    // force reuse of the existing prebuilt module files instead.
    PrebuiltModuleVFSMapT PrebuiltModuleVFSMap;
    if (!ScanInstance.getPreprocessorOpts().ImplicitPCHInclude.empty())
      if (visitPrebuiltModule(
              ScanInstance.getPreprocessorOpts().ImplicitPCHInclude,
              ScanInstance,
              ScanInstance.getHeaderSearchOpts().PrebuiltModuleFiles,
              PrebuiltModuleVFSMap, ScanInstance.getDiagnostics()))
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
      Opts->Targets = {
          deduceDepTarget(ScanInstance.getFrontendOpts().OutputFile,
                          ScanInstance.getFrontendOpts().Inputs)};
    if (Service.getFormat() == ScanningOutputFormat::Make) {
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
    switch (Service.getFormat()) {
    case ScanningOutputFormat::Make:
    case ScanningOutputFormat::Tree:
      ScanInstance.addDependencyCollector(
          std::make_shared<DependencyConsumerForwarder>(
              std::move(Opts), WorkingDirectory, Consumer, EmitDependencyFile));
      break;
    case ScanningOutputFormat::IncludeTree:
    case ScanningOutputFormat::P1689:
    case ScanningOutputFormat::Full:
    case ScanningOutputFormat::FullTree:
    case ScanningOutputFormat::FullIncludeTree:
      if (EmitDependencyFile) {
        auto DFG =
            std::make_shared<ReversePrefixMappingDependencyFileGenerator>(
                *Opts);
        DFG->initialize(ScanInstance.getInvocation());
        ScanInstance.addDependencyCollector(std::move(DFG));
      }

      MDC = std::make_shared<ModuleDepCollector>(
          Service, std::move(Opts), ScanInstance, Consumer, Controller,
          OriginalInvocation, std::move(PrebuiltModuleVFSMap));
      ScanInstance.addDependencyCollector(MDC);
      ScanInstance.setGenModuleActionWrapper(
          [&Controller = Controller](const FrontendOptions &Opts,
                                     std::unique_ptr<FrontendAction> Wrapped) {
            return std::make_unique<WrapScanModuleBuildAction>(
                std::move(Wrapped), Controller);
          });
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
    ScanInstance.getHeaderSearchOpts().ModulesSkipPragmaDiagnosticMappings =
        true;

    // Avoid some checks and module map parsing when loading PCM files.
    ScanInstance.getPreprocessorOpts().ModulesCheckRelocated = false;

    std::unique_ptr<FrontendAction> Action;

    if (Service.getFormat() == ScanningOutputFormat::P1689)
      Action = std::make_unique<PreprocessOnlyAction>();
    else if (ModuleName)
      Action = std::make_unique<GetDependenciesByModuleNameAction>(*ModuleName);
    else
      Action = std::make_unique<ReadPCHAndPreprocessAction>();

    // Normally this would be handled by GeneratePCHAction
    if (ScanInstance.getFrontendOpts().ProgramAction == frontend::GeneratePCH)
      ScanInstance.getLangOpts().CompilingPCH = true;

    if (Error E = Controller.initialize(ScanInstance, OriginalInvocation))
      return reportError(std::move(E));

    if (ScanInstance.getDiagnostics().hasErrorOccurred())
      return false;

    // ExecuteAction is responsible for calling finish.
    DiagConsumerFinished = true;

    if (!ScanInstance.ExecuteAction(*Action))
      return false;

    if (MDC)
      MDC->applyDiscoveredDependencies(OriginalInvocation);

    if (Error E = Controller.finalize(ScanInstance, OriginalInvocation))
      return reportError(std::move(E));

    // Forward any CAS results to consumer.
    std::string ID = OriginalInvocation.getFileSystemOpts().CASFileSystemRootID;
    if (!ID.empty())
      Consumer.handleCASFileSystemRootID(std::move(ID));
    ID = OriginalInvocation.getFrontendOpts().CASIncludeTreeID;
    if (!ID.empty())
      Consumer.handleIncludeTreeID(std::move(ID));

    LastCC1Arguments = OriginalInvocation.getCC1CommandLine();
    LastCC1CacheKey = Controller.getCacheKey(OriginalInvocation);

    // Propagate the statistics to the parent FileManager.
    DriverFileMgr->AddStats(ScanInstance.getFileManager());

    return true;
  }

  bool hasScanned() const { return Scanned; }
  bool hasDiagConsumerFinished() const { return DiagConsumerFinished; }

  /// Take the cc1 arguments corresponding to the most recent invocation used
  /// with this action. Any modifications implied by the discovered dependencies
  /// will have already been applied.
  std::vector<std::string> takeLastCC1Arguments() {
    std::vector<std::string> Result;
    std::swap(Result, LastCC1Arguments); // Reset LastCC1Arguments to empty.
    return Result;
  }

  std::optional<std::string> takeLastCC1CacheKey() {
    std::optional<std::string> Result;
    std::swap(Result, LastCC1CacheKey);
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

  DependencyScanningService &Service;
  StringRef WorkingDirectory;
  DependencyConsumer &Consumer;
  DependencyActionController &Controller;
  llvm::IntrusiveRefCntPtr<DependencyScanningWorkerFilesystem> DepFS;
  llvm::IntrusiveRefCntPtr<DependencyScanningCASFilesystem> DepCASFS;
  llvm::IntrusiveRefCntPtr<llvm::cas::CachingOnDiskFileSystem> CacheFS;
  bool DisableFree;
  const CASOptions &CASOpts;
  bool EmitDependencyFile = false;
  bool DiagGenerationAsCompilation;
  std::optional<StringRef> ModuleName;
  std::optional<CompilerInstance> ScanInstanceStorage;
  std::shared_ptr<ModuleDepCollector> MDC;
  std::vector<std::string> LastCC1Arguments;
  std::optional<std::string> LastCC1CacheKey;
  bool Scanned = false;
  bool DiagConsumerFinished = false;
  raw_ostream *VerboseOS;
};

} // end anonymous namespace

DependencyScanningWorker::DependencyScanningWorker(
    DependencyScanningService &Service,
    llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS)
    : Service(Service),
      CASOpts(Service.getCASOpts()), CAS(Service.getCAS()) {
  PCHContainerOps = std::make_shared<PCHContainerOperations>();
  // We need to read object files from PCH built outside the scanner.
  PCHContainerOps->registerReader(
      std::make_unique<ObjectFilePCHContainerReader>());
  // The scanner itself writes only raw ast files.
  PCHContainerOps->registerWriter(std::make_unique<RawPCHContainerWriter>());

  if (Service.shouldTraceVFS())
    FS = llvm::makeIntrusiveRefCnt<llvm::vfs::TracingFileSystem>(std::move(FS));

  if (Service.useCASFS()) {
    CacheFS = Service.getSharedFS().createProxyFS();
    DepCASFS = new DependencyScanningCASFilesystem(CacheFS, *Service.getCache());
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

static std::unique_ptr<DiagnosticOptions>
createDiagOptions(const std::vector<std::string> &CommandLine) {
  std::vector<const char *> CLI;
  for (const std::string &Arg : CommandLine)
    CLI.push_back(Arg.c_str());
  auto DiagOpts = CreateAndPopulateDiagOpts(CLI);
  sanitizeDiagOpts(*DiagOpts);
  return DiagOpts;
}

llvm::Error DependencyScanningWorker::computeDependencies(
    StringRef WorkingDirectory, const std::vector<std::string> &CommandLine,
    DependencyConsumer &Consumer, DependencyActionController &Controller,
    std::optional<llvm::MemoryBufferRef> TUBuffer) {
  // Capture the emitted diagnostics and report them to the client
  // in the case of a failure.
  std::string DiagnosticOutput;
  llvm::raw_string_ostream DiagnosticsOS(DiagnosticOutput);
  auto DiagOpts = createDiagOptions(CommandLine);
  TextDiagnosticPrinter DiagPrinter(DiagnosticsOS, DiagOpts.release());

  if (computeDependencies(WorkingDirectory, CommandLine, Consumer, Controller,
                          DiagPrinter, TUBuffer))
    return llvm::Error::success();
  return llvm::make_error<llvm::StringError>(DiagnosticsOS.str(),
                                             llvm::inconvertibleErrorCode());
}

llvm::Error DependencyScanningWorker::computeDependencies(
    StringRef WorkingDirectory, const std::vector<std::string> &CommandLine,
    DependencyConsumer &Consumer, DependencyActionController &Controller,
    StringRef ModuleName) {
  // Capture the emitted diagnostics and report them to the client
  // in the case of a failure.
  std::string DiagnosticOutput;
  llvm::raw_string_ostream DiagnosticsOS(DiagnosticOutput);
  auto DiagOpts = createDiagOptions(CommandLine);
  TextDiagnosticPrinter DiagPrinter(DiagnosticsOS, DiagOpts.release());

  if (computeDependencies(WorkingDirectory, CommandLine, Consumer, Controller,
                          DiagPrinter, ModuleName))
    return llvm::Error::success();
  return llvm::make_error<llvm::StringError>(DiagnosticsOS.str(),
                                             llvm::inconvertibleErrorCode());
}

static bool forEachDriverJob(
    ArrayRef<std::string> ArgStrs, DiagnosticsEngine &Diags, FileManager &FM,
    llvm::function_ref<bool(const driver::Command &Cmd)> Callback) {
  SmallVector<const char *, 256> Argv;
  Argv.reserve(ArgStrs.size());
  for (const std::string &Arg : ArgStrs)
    Argv.push_back(Arg.c_str());

  llvm::vfs::FileSystem *FS = &FM.getVirtualFileSystem();

  std::unique_ptr<driver::Driver> Driver = std::make_unique<driver::Driver>(
      Argv[0], llvm::sys::getDefaultTargetTriple(), Diags,
      "clang LLVM compiler", FS);
  Driver->setTitle("clang_based_tool");

  llvm::BumpPtrAllocator Alloc;
  bool CLMode = driver::IsClangCL(
      driver::getDriverMode(Argv[0], ArrayRef(Argv).slice(1)));

  if (llvm::Error E = driver::expandResponseFiles(Argv, CLMode, Alloc, FS)) {
    Diags.Report(diag::err_drv_expand_response_file)
        << llvm::toString(std::move(E));
    return false;
  }

  const std::unique_ptr<driver::Compilation> Compilation(
      Driver->BuildCompilation(llvm::ArrayRef(Argv)));
  if (!Compilation)
    return false;

  if (Compilation->containsError())
    return false;

  for (const driver::Command &Job : Compilation->getJobs()) {
    if (!Callback(Job))
      return false;
  }
  return true;
}

static bool createAndRunToolInvocation(
    std::vector<std::string> CommandLine, DependencyScanningAction &Action,
    FileManager &FM,
    std::shared_ptr<clang::PCHContainerOperations> &PCHContainerOps,
    DiagnosticsEngine &Diags, DependencyConsumer &Consumer) {

  // Save executable path before providing CommandLine to ToolInvocation
  std::string Executable = CommandLine[0];
  ToolInvocation Invocation(std::move(CommandLine), &Action, &FM,
                            PCHContainerOps);
  Invocation.setDiagnosticConsumer(Diags.getClient());
  Invocation.setDiagnosticOptions(&Diags.getDiagnosticOptions());
  if (!Invocation.run())
    return false;

  std::vector<std::string> Args = Action.takeLastCC1Arguments();
  std::optional<std::string> CacheKey = Action.takeLastCC1CacheKey();
  Consumer.handleBuildCommand(
      {std::move(Executable), std::move(Args), std::move(CacheKey)});
  return true;
}

bool DependencyScanningWorker::scanDependencies(
    StringRef WorkingDirectory, const std::vector<std::string> &CommandLine,
    DependencyConsumer &Consumer, DependencyActionController &Controller,
    DiagnosticConsumer &DC, llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS,
    std::optional<StringRef> ModuleName) {
  auto FileMgr =
      llvm::makeIntrusiveRefCnt<FileManager>(FileSystemOptions{}, FS);

  std::vector<const char *> CCommandLine(CommandLine.size(), nullptr);
  llvm::transform(CommandLine, CCommandLine.begin(),
                  [](const std::string &Str) { return Str.c_str(); });
  auto DiagOpts = CreateAndPopulateDiagOpts(CCommandLine);
  sanitizeDiagOpts(*DiagOpts);
  IntrusiveRefCntPtr<DiagnosticsEngine> Diags =
      CompilerInstance::createDiagnostics(FileMgr->getVirtualFileSystem(),
                                          DiagOpts.release(), &DC,
                                          /*ShouldOwnClient=*/false);

  // Although `Diagnostics` are used only for command-line parsing, the
  // custom `DiagConsumer` might expect a `SourceManager` to be present.
  SourceManager SrcMgr(*Diags, *FileMgr);
  Diags->setSourceManager(&SrcMgr);
  // DisableFree is modified by Tooling for running
  // in-process; preserve the original value, which is
  // always true for a driver invocation.
  bool DisableFree = true;
  DependencyScanningAction Action(Service, WorkingDirectory, Consumer, Controller, DepFS,
                                  DepCASFS, CacheFS,
                                  DisableFree,
                                  /*EmitDependencyFile=*/false,
                                  /*DiagGenerationAsCompilation=*/false, getCASOpts(),
                                  ModuleName);
  bool Success = false;
  if (CommandLine[1] == "-cc1") {
    Success = createAndRunToolInvocation(CommandLine, Action, *FileMgr,
                                         PCHContainerOps, *Diags, Consumer);
  } else {
    Success = forEachDriverJob(
        CommandLine, *Diags, *FileMgr, [&](const driver::Command &Cmd) {
          if (StringRef(Cmd.getCreator().getName()) != "clang") {
            // Non-clang command. Just pass through to the dependency
            // consumer.
            Consumer.handleBuildCommand(
                {Cmd.getExecutable(),
                 {Cmd.getArguments().begin(), Cmd.getArguments().end()},
                 {}});
            return true;
          }

          // Insert -cc1 comand line options into Argv
          std::vector<std::string> Argv;
          Argv.push_back(Cmd.getExecutable());
          Argv.insert(Argv.end(), Cmd.getArguments().begin(),
                      Cmd.getArguments().end());

          // Create an invocation that uses the underlying file
          // system to ensure that any file system requests that
          // are made by the driver do not go through the
          // dependency scanning filesystem.
          return createAndRunToolInvocation(std::move(Argv), Action, *FileMgr,
                                            PCHContainerOps, *Diags, Consumer);
        });
  }

  if (Success && !Action.hasScanned())
    Diags->Report(diag::err_fe_expected_compiler_job)
        << llvm::join(CommandLine, " ");

  // Ensure finish() is called even if we never reached ExecuteAction().
  if (!Action.hasDiagConsumerFinished())
    DC.finish();

  return Success && Action.hasScanned();
}

bool DependencyScanningWorker::computeDependencies(
    StringRef WorkingDirectory, const std::vector<std::string> &CommandLine,
    DependencyConsumer &Consumer, DependencyActionController &Controller,
    DiagnosticConsumer &DC, std::optional<llvm::MemoryBufferRef> TUBuffer) {
  // Reset what might have been modified in the previous worker invocation.
  BaseFS->setCurrentWorkingDirectory(WorkingDirectory);

  std::optional<std::vector<std::string>> ModifiedCommandLine;
  llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> ModifiedFS;

  // If we're scanning based on a module name alone, we don't expect the client
  // to provide us with an input file. However, the driver really wants to have
  // one. Let's just make it up to make the driver happy.
  if (TUBuffer) {
    auto OverlayFS =
        llvm::makeIntrusiveRefCnt<llvm::vfs::OverlayFileSystem>(BaseFS);
    auto InMemoryFS =
        llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
    InMemoryFS->setCurrentWorkingDirectory(WorkingDirectory);
    auto InputPath = TUBuffer->getBufferIdentifier();
    InMemoryFS->addFile(
        InputPath, 0,
        llvm::MemoryBuffer::getMemBufferCopy(TUBuffer->getBuffer()));
    llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> InMemoryOverlay =
        InMemoryFS;

    // If we are using a CAS but not dependency CASFS, we need to provide the
    // fake input file in a CASProvidingFS for include-tree.
    if (CAS && !DepCASFS)
      InMemoryOverlay =
          llvm::cas::createCASProvidingFileSystem(CAS, std::move(InMemoryFS));

    OverlayFS->pushOverlay(InMemoryOverlay);
    ModifiedFS = OverlayFS;
    ModifiedCommandLine = CommandLine;
    ModifiedCommandLine->emplace_back(InputPath);
  }

  const std::vector<std::string> &FinalCommandLine =
      ModifiedCommandLine ? *ModifiedCommandLine : CommandLine;
  auto &FinalFS = ModifiedFS ? ModifiedFS : BaseFS;

  return scanDependencies(WorkingDirectory, FinalCommandLine, Consumer,
                          Controller, DC, FinalFS, /*ModuleName=*/std::nullopt);
}

bool DependencyScanningWorker::computeDependencies(
    StringRef WorkingDirectory, const std::vector<std::string> &CommandLine,
    DependencyConsumer &Consumer, DependencyActionController &Controller,
    DiagnosticConsumer &DC, StringRef ModuleName) {
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
  llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> InMemoryOverlay = InMemoryFS;

  // If we are using a CAS but not dependency CASFS, we need to provide the
  // fake input file in a CASProvidingFS for include-tree.
  if (CAS && !DepCASFS)
    InMemoryOverlay =
        llvm::cas::createCASProvidingFileSystem(CAS, std::move(InMemoryFS));

  OverlayFS->pushOverlay(InMemoryOverlay);
  auto ModifiedCommandLine = CommandLine;
  ModifiedCommandLine.emplace_back(FakeInputPath);

  return scanDependencies(WorkingDirectory, ModifiedCommandLine, Consumer,
                          Controller, DC, OverlayFS, ModuleName);
}

DependencyActionController::~DependencyActionController() {}

void DependencyScanningWorker::computeDependenciesFromCompilerInvocation(
    std::shared_ptr<CompilerInvocation> Invocation, StringRef WorkingDirectory,
    DependencyConsumer &DepsConsumer, DependencyActionController &Controller,
    DiagnosticConsumer &DiagsConsumer, raw_ostream *VerboseOS,
    bool DiagGenerationAsCompilation) {
  BaseFS->setCurrentWorkingDirectory(WorkingDirectory);

  // Adjust the invocation.
  auto &Frontend = Invocation->getFrontendOpts();
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
  DependencyScanningAction Action(Service, WorkingDirectory, DepsConsumer,
                                  Controller, DepFS, DepCASFS, CacheFS,
                                  /*DisableFree=*/false,
                                  /*EmitDependencyFile=*/!DepFile.empty(),
                                  DiagGenerationAsCompilation, getCASOpts(),
                                  /*ModuleName=*/std::nullopt, VerboseOS);

  // Ignore result; we're just collecting dependencies.
  //
  // FIXME: will clients other than -cc1scand care?
  IntrusiveRefCntPtr<FileManager> ActiveFiles =
      new FileManager(Invocation->getFileSystemOpts(), BaseFS);
  (void)Action.runInvocation(std::move(Invocation), ActiveFiles.get(),
                             PCHContainerOps, &DiagsConsumer);
}
