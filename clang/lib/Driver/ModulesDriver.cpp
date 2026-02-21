//===--- ModulesDriver.cpp - Driver managed module builds -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines functionality to support driver managed builds for
/// compilations which use Clang modules or standard C++20 named modules.
///
//===----------------------------------------------------------------------===//

#include "clang/Driver/ModulesDriver.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/LLVM.h"
#include "clang/DependencyScanning/DependencyScanningUtils.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Job.h"
#include "clang/Driver/Tool.h"
#include "clang/Driver/ToolChain.h"
#include "clang/Frontend/StandaloneDiagnostic.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/DirectedGraph.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/ThreadPool.h"
#include <algorithm>
#include <utility>

namespace deps = clang::dependencies;

using namespace llvm::opt;
using namespace clang;
using namespace driver;
using namespace modules;

namespace clang::driver::modules {
static bool fromJSON(const llvm::json::Value &Params,
                     StdModuleManifest::Module::LocalArguments &LocalArgs,
                     llvm::json::Path P) {
  llvm::json::ObjectMapper O(Params, P);
  return O.mapOptional("system-include-directories",
                       LocalArgs.SystemIncludeDirs);
}

static bool fromJSON(const llvm::json::Value &Params,
                     StdModuleManifest::Module &ModuleEntry,
                     llvm::json::Path P) {
  llvm::json::ObjectMapper O(Params, P);
  return O.map("is-std-library", ModuleEntry.IsStdlib) &&
         O.map("logical-name", ModuleEntry.LogicalName) &&
         O.map("source-path", ModuleEntry.SourcePath) &&
         O.mapOptional("local-arguments", ModuleEntry.LocalArgs);
}

static bool fromJSON(const llvm::json::Value &Params,
                     StdModuleManifest &Manifest, llvm::json::Path P) {
  llvm::json::ObjectMapper O(Params, P);
  return O.map("modules", Manifest.Modules);
}
} // namespace clang::driver::modules

/// Parses the Standard library module manifest from \p Buffer.
static Expected<StdModuleManifest> parseManifest(StringRef Buffer) {
  auto ParsedOrErr = llvm::json::parse(Buffer);
  if (!ParsedOrErr)
    return ParsedOrErr.takeError();

  StdModuleManifest Manifest;
  llvm::json::Path::Root Root;
  if (!fromJSON(*ParsedOrErr, Manifest, Root))
    return Root.getError();

  return Manifest;
}

/// Converts each file path in manifest from relative to absolute.
///
/// Each file path in the manifest is expected to be relative the manifest's
/// location \p ManifestPath itself.
static void makeManifestPathsAbsolute(
    MutableArrayRef<StdModuleManifest::Module> ManifestEntries,
    StringRef ManifestPath) {
  StringRef ManifestDir = llvm::sys::path::parent_path(ManifestPath);
  SmallString<256> TempPath;

  auto PrependManifestDir = [&](StringRef Path) {
    TempPath = ManifestDir;
    llvm::sys::path::append(TempPath, Path);
    return std::string(TempPath);
  };

  for (auto &Entry : ManifestEntries) {
    Entry.SourcePath = PrependManifestDir(Entry.SourcePath);
    if (!Entry.LocalArgs)
      continue;

    for (auto &IncludeDir : Entry.LocalArgs->SystemIncludeDirs)
      IncludeDir = PrependManifestDir(IncludeDir);
  }
}

Expected<StdModuleManifest>
driver::modules::readStdModuleManifest(StringRef ManifestPath,
                                       llvm::vfs::FileSystem &VFS) {
  auto MemBufOrErr = VFS.getBufferForFile(ManifestPath);
  if (!MemBufOrErr)
    return llvm::createFileError(ManifestPath, MemBufOrErr.getError());

  auto ManifestOrErr = parseManifest((*MemBufOrErr)->getBuffer());
  if (!ManifestOrErr)
    return ManifestOrErr.takeError();
  auto Manifest = std::move(*ManifestOrErr);

  makeManifestPathsAbsolute(Manifest.Modules, ManifestPath);
  return Manifest;
}

void driver::modules::buildStdModuleManifestInputs(
    ArrayRef<StdModuleManifest::Module> ManifestEntries, Compilation &C,
    InputList &Inputs) {
  DerivedArgList &Args = C.getArgs();
  const OptTable &Opts = C.getDriver().getOpts();
  for (const auto &Entry : ManifestEntries) {
    auto *InputArg =
        makeInputArg(Args, Opts, Args.MakeArgString(Entry.SourcePath));
    Inputs.emplace_back(types::TY_CXXModule, InputArg);
  }
}

/// Computes the -fmodule-cache-path for this compilation.
static std::optional<std::string>
getModuleCachePath(llvm::opt::DerivedArgList &Args) {
  if (const Arg *A = Args.getLastArg(options::OPT_fmodules_cache_path))
    return A->getValue();

  if (SmallString<128> Path; Driver::getDefaultModuleCachePath(Path))
    return std::string(Path);

  return std::nullopt;
}

using ManifestEntryLookup =
    llvm::DenseMap<StringRef, const StdModuleManifest::Module *>;

/// Builds a mapping from a module's source path to its entry in the manifest.
static ManifestEntryLookup
buildManifestLookupMap(ArrayRef<StdModuleManifest::Module> ManifestEntries) {
  ManifestEntryLookup ManifestEntryBySource;
  for (auto &Entry : ManifestEntries) {
    const bool Inserted =
        ManifestEntryBySource.try_emplace(Entry.SourcePath, &Entry).second;
    assert(Inserted &&
           "Manifest defines multiple modules with the same source path.");
  }
  return ManifestEntryBySource;
}

/// Returns the manifest entry corresponding to \p Job, or \c nullptr if none
/// exists.
static const StdModuleManifest::Module *
getManifestEntryForCommand(const Command &Job,
                           const ManifestEntryLookup &ManifestLookup) {
  for (const auto &II : Job.getInputInfos()) {
    if (const auto It = ManifestLookup.find(II.getFilename());
        It != ManifestLookup.end())
      return It->second;
  }
  return nullptr;
}

/// Adds all \p SystemIncludeDirs to \p Job's arguments.
static void
addSystemIncludeDirsFromManifest(Compilation &C, Command &Job,
                                 ArrayRef<std::string> SystemIncludeDirs) {
  const ToolChain &TC = Job.getCreator().getToolChain();
  const DerivedArgList &TCArgs =
      C.getArgsForToolChain(&TC, Job.getSource().getOffloadingArch(),
                            Job.getSource().getOffloadingDeviceKind());

  ArgStringList NewArgs = Job.getArguments();
  for (const auto &IncludeDir : SystemIncludeDirs)
    TC.addSystemInclude(TCArgs, NewArgs, IncludeDir);
  Job.replaceArguments(NewArgs);
}

static bool isCC1Job(const Command &Job) {
  return StringRef(Job.getCreator().getName()) == "clang";
}

/// For each job that generates a Standard library module, applies any
/// local arguments specified in the corresponding module manifest entry.
static void
applyLocalArgsFromManifest(Compilation &C,
                           const ManifestEntryLookup &ManifestLookup,
                           MutableArrayRef<std::unique_ptr<Command>> Jobs) {
  for (auto &Job : Jobs) {
    if (!isCC1Job(*Job))
      continue;

    const auto *Entry = getManifestEntryForCommand(*Job, ManifestLookup);
    if (!Entry || !Entry->LocalArgs)
      continue;

    const auto &IncludeDirs = Entry->LocalArgs->SystemIncludeDirs;
    addSystemIncludeDirsFromManifest(C, *Job, IncludeDirs);
  }
}

/// Returns true if a dependency scan can be performed using \p Job.
static bool isDependencyScannableJob(const Command &Job) {
  if (!isCC1Job(Job))
    return false;
  const auto &InputInfos = Job.getInputInfos();
  return !InputInfos.empty() && types::isSrcFile(InputInfos.front().getType());
}

namespace {
/// Pool of reusable dependency scanning workers and their contexts with
/// RAII-based acquire/release.
class ScanningWorkerPool {
public:
  ScanningWorkerPool(size_t NumWorkers,
                     deps::DependencyScanningService &ScanningService) {
    for (size_t I = 0; I < NumWorkers; ++I)
      Slots.emplace_back(ScanningService);

    AvailableSlots.resize(NumWorkers);
    std::iota(AvailableSlots.begin(), AvailableSlots.end(), 0);
  }

  /// Acquires a unique pointer to a dependency scanning worker and its
  /// context.
  ///
  /// The worker bundle automatically released back to the pool when the
  /// pointer is destroyed. The pool has to outlive the leased worker bundle.
  [[nodiscard]] auto scopedAcquire() {
    std::unique_lock<std::mutex> UL(Lock);
    CV.wait(UL, [&] { return !AvailableSlots.empty(); });
    const size_t Index = AvailableSlots.pop_back_val();
    auto ReleaseHandle = [this, Index](WorkerBundle *) { release(Index); };
    return std::unique_ptr<WorkerBundle, decltype(ReleaseHandle)>(
        &Slots[Index], ReleaseHandle);
  }

private:
  /// Releases the worker bundle at \c Index back into the pool.
  void release(size_t Index) {
    {
      std::scoped_lock<std::mutex> SL(Lock);
      AvailableSlots.push_back(Index);
    }
    CV.notify_one();
  }

  /// A scanning worker with its associated context.
  struct WorkerBundle {
    WorkerBundle(deps::DependencyScanningService &ScanningService)
        : Worker(std::make_unique<deps::DependencyScanningWorker>(
              ScanningService)) {}

    std::unique_ptr<deps::DependencyScanningWorker> Worker;
    llvm::DenseSet<deps::ModuleID> SeenModules;
  };

  std::mutex Lock;
  std::condition_variable CV;
  SmallVector<size_t> AvailableSlots;
  SmallVector<WorkerBundle, 0> Slots;
};
} // anonymous namespace

// Creates a ThreadPool and a corresponding ScanningWorkerPool optimized for
// the configuration of dependency scan inputs.
static std::pair<std::unique_ptr<llvm::ThreadPoolInterface>,
                 std::unique_ptr<ScanningWorkerPool>>
createOptimalThreadAndWorkerPool(
    size_t NumScanInputs, bool HasStdlibModuleInputs,
    deps::DependencyScanningService &ScanningService) {
  // TODO: Benchmark: Determine the optimal number of worker threads for a
  // given number of inputs. How many inputs are required for multi-threading
  // to be beneficial? How many inputs should each thread scan at least?
#if LLVM_ENABLE_THREADS
  std::unique_ptr<llvm::ThreadPoolInterface> ThreadPool;
  size_t WorkerCount;

  if (NumScanInputs == 1 || (HasStdlibModuleInputs && NumScanInputs <= 2)) {
    auto S = llvm::optimal_concurrency(1);
    ThreadPool = std::make_unique<llvm::SingleThreadExecutor>(std::move(S));
    WorkerCount = 1;
  } else {
    auto ThreadPoolStrategy = llvm::optimal_concurrency(
        NumScanInputs - static_cast<size_t>(HasStdlibModuleInputs));
    ThreadPool = std::make_unique<llvm::DefaultThreadPool>(
        std::move(ThreadPoolStrategy));
    const size_t MaxConcurrency = ThreadPool->getMaxConcurrency();
    const size_t MaxConcurrentlyScannedInputs =
        NumScanInputs -
        (HasStdlibModuleInputs && NumScanInputs < MaxConcurrency ? 1 : 0);
    WorkerCount = std::min(MaxConcurrency, MaxConcurrentlyScannedInputs);
  }
#else
  auto ThreadPool = std::make_unique<llvm::SingleThreadExecutor>();
  size_t WorkerCount = 1;
#endif

  return {std::move(ThreadPool),
          std::make_unique<ScanningWorkerPool>(WorkerCount, ScanningService)};
}

static StringRef getTriple(const Command &Job) {
  return Job.getCreator().getToolChain().getTriple().getTriple();
}

using ModuleNameAndTriple = std::pair<StringRef, StringRef>;

namespace {
/// Thread-safe registry of Standard library scan inputs.
struct StdlibModuleScanScheduler {
  StdlibModuleScanScheduler(const llvm::DenseMap<ModuleNameAndTriple, size_t>
                                &StdlibModuleScanIndexByID)
      : StdlibModuleScanIndexByID(StdlibModuleScanIndexByID) {
    ScheduledScanInputs.reserve(StdlibModuleScanIndexByID.size());
  }

  /// Returns the indices of scan inputs corresponding to newly imported
  /// Standard library modules.
  ///
  /// Thread-safe.
  SmallVector<size_t, 2> getNewScanInputs(ArrayRef<std::string> NamedModuleDeps,
                                          StringRef Triple) {
    SmallVector<size_t, 2> NewScanInputs;
    std::scoped_lock<std::mutex> Guard(Lock);
    for (const auto &ModuleName : NamedModuleDeps) {
      const auto It = StdlibModuleScanIndexByID.find({ModuleName, Triple});
      if (It == StdlibModuleScanIndexByID.end())
        continue;
      const size_t ScanIndex = It->second;
      const bool AlreadyScheduled =
          !ScheduledScanInputs.insert(ScanIndex).second;
      if (AlreadyScheduled)
        continue;
      NewScanInputs.push_back(ScanIndex);
    }
    return NewScanInputs;
  }

private:
  const llvm::DenseMap<ModuleNameAndTriple, size_t> &StdlibModuleScanIndexByID;
  llvm::SmallDenseSet<size_t> ScheduledScanInputs;
  std::mutex Lock;
};

/// Collects diagnostics in a form that can be retained until after their
/// associated SourceManager is destroyed.
class StandaloneDiagCollector : public DiagnosticConsumer {
public:
  void BeginSourceFile(const LangOptions &LangOpts,
                       const Preprocessor *PP = nullptr) override {
    this->LangOpts = &LangOpts;
  }

  void HandleDiagnostic(DiagnosticsEngine::Level Level,
                        const Diagnostic &Info) override {
    StoredDiagnostic StoredDiag(Level, Info);
    StandaloneDiags.emplace_back(*LangOpts, StoredDiag);
    DiagnosticConsumer::HandleDiagnostic(Level, Info);
  }

  SmallVector<StandaloneDiagnostic, 0> takeDiagnostics() {
    return std::move(StandaloneDiags);
  }

private:
  const LangOptions *LangOpts = nullptr;
  SmallVector<StandaloneDiagnostic, 0> StandaloneDiags;
};

/// RAII utility to report collected StandaloneDiagnostic through a
/// DiagnosticsEngine.
///
/// The driver's DiagnosticsEngine usually does not have a SourceManager at
/// this point of building the compilation, in which case the
/// StandaloneDiagReporter supplies its own.
class StandaloneDiagReporter {
public:
  explicit StandaloneDiagReporter(DiagnosticsEngine &Diags) : Diags(Diags) {
    if (!Diags.hasSourceManager()) {
      FileSystemOptions Opts;
      Opts.WorkingDir = ".";
      OwnedFileMgr = llvm::makeIntrusiveRefCnt<FileManager>(std::move(Opts));
      OwnedSrcMgr =
          llvm::makeIntrusiveRefCnt<SourceManager>(Diags, *OwnedFileMgr);
    }
  }

  /// Emits all diagnostics in \c StandaloneDiags using the associated
  /// DiagnosticsEngine.
  void Report(ArrayRef<StandaloneDiagnostic> StandaloneDiags) const {
    llvm::StringMap<SourceLocation> SrcLocCache;
    Diags.getClient()->BeginSourceFile(LangOptions(), nullptr);
    for (auto &StandaloneDiag : StandaloneDiags) {
      const auto StoredDiag = translateStandaloneDiag(
          getFileManager(), getSourceManager(), StandaloneDiag, SrcLocCache);
      Diags.Report(StoredDiag);
    }
    Diags.getClient()->EndSourceFile();
  }

private:
  DiagnosticsEngine &Diags;
  IntrusiveRefCntPtr<FileManager> OwnedFileMgr;
  IntrusiveRefCntPtr<SourceManager> OwnedSrcMgr;

  FileManager &getFileManager() const {
    if (OwnedFileMgr)
      return *OwnedFileMgr;
    return Diags.getSourceManager().getFileManager();
  }

  SourceManager &getSourceManager() const {
    if (OwnedSrcMgr)
      return *OwnedSrcMgr;
    return Diags.getSourceManager();
  }
};
} // anonymous namespace

/// Report the diagnostics collected during each dependency scan.
static void reportAllScanDiagnostics(
    SmallVectorImpl<SmallVector<StandaloneDiagnostic, 0>> &&AllScanDiags,
    DiagnosticsEngine &Diags) {
  StandaloneDiagReporter Reporter(Diags);
  for (auto &SingleScanDiags : AllScanDiags)
    Reporter.Report(SingleScanDiags);
}

/// Construct a path for the explicitly built PCM.
static std::string constructPCMPath(const deps::ModuleID &ID,
                                    StringRef OutputDir) {
  assert(!ID.ModuleName.empty() && !ID.ContextHash.empty() &&
         "Invalid ModuleID!");
  SmallString<256> ExplicitPCMPath(OutputDir);
  llvm::sys::path::append(ExplicitPCMPath, ID.ContextHash,
                          ID.ModuleName + "-" + ID.ContextHash + ".pcm");
  return std::string(ExplicitPCMPath);
}

namespace {
/// A simple dependency action controller that only provides module lookup for
/// Clang modules.
class ModuleLookupController : public deps::DependencyActionController {
public:
  ModuleLookupController(StringRef OutputDir) : OutputDir(OutputDir) {}

  std::string lookupModuleOutput(const deps::ModuleDeps &MD,
                                 deps::ModuleOutputKind Kind) override {
    if (Kind == deps::ModuleOutputKind::ModuleFile)
      return constructPCMPath(MD.ID, OutputDir);

    // Driver command lines that trigger lookups for unsupported
    // ModuleOutputKinds are not supported by the modules driver. Those
    // command lines should probably be adjusted or rejected in
    // Driver::handleArguments or Driver::HandleImmediateArgs.
    llvm::reportFatalInternalError(
        "call to lookupModuleOutput with unexpected ModuleOutputKind");
  }

private:
  StringRef OutputDir;
};

/// The full dependencies for a specific command-line input.
struct InputDependencies {
  /// The name of the C++20 module provided by this translation unit.
  std::string ModuleName;

  /// A list of modules this translation unit directly depends on, not including
  /// transitive dependencies.
  ///
  /// This may include modules with a different context hash when it can be
  /// determined that the differences are benign for this compilation.
  std::vector<deps::ModuleID> ClangModuleDeps;

  /// A list of the C++20 named modules this translation unit depends on.
  ///
  /// All C++20 named module dependencies are expected to target the same triple
  /// as this translation unit.
  std::vector<std::string> NamedModuleDeps;

  /// A collection of absolute paths to files that this translation unit
  /// directly depends on, not including transitive dependencies.
  std::vector<std::string> FileDeps;

  /// The compiler invocation with modifications to properly import all Clang
  /// module dependencies. Does not include argv[0].
  std::vector<std::string> BuildArgs;
};
} // anonymous namespace

static InputDependencies makeInputDeps(deps::TranslationUnitDeps &&TUDeps) {
  InputDependencies InputDeps;
  InputDeps.ModuleName = std::move(TUDeps.ID.ModuleName);
  InputDeps.NamedModuleDeps = std::move(TUDeps.NamedModuleDeps);
  InputDeps.ClangModuleDeps = std::move(TUDeps.ClangModuleDeps);
  InputDeps.FileDeps = std::move(TUDeps.FileDeps);
  assert(TUDeps.Commands.size() == 1 && "Expected exactly one command");
  InputDeps.BuildArgs = std::move(TUDeps.Commands.front().Arguments);
  return InputDeps;
}

/// Constructs the full command line, including the executable, for \p Job.
static SmallVector<std::string, 0> buildCommandLine(const Command &Job) {
  const auto &JobArgs = Job.getArguments();
  SmallVector<std::string, 0> CommandLine;
  CommandLine.reserve(JobArgs.size() + 1);
  CommandLine.emplace_back(Job.getExecutable());
  for (const char *Arg : JobArgs)
    CommandLine.emplace_back(Arg);
  return CommandLine;
}

/// Performs a dependency scan for a single job.
///
/// \returns a pair containing TranslationUnitDeps on success, or std::nullopt
/// on failure, along with any diagnostics produced.
static std::pair<std::optional<deps::TranslationUnitDeps>,
                 SmallVector<StandaloneDiagnostic, 0>>
scanDependenciesForJob(const Command &Job, ScanningWorkerPool &WorkerPool,
                       ModuleLookupController &LookupController) {
  StandaloneDiagCollector DiagConsumer;
  std::optional<deps::TranslationUnitDeps> MaybeTUDeps;

  {
    const auto CC1CommandLine = buildCommandLine(Job);
    auto WorkerBundleHandle = WorkerPool.scopedAcquire();
    deps::FullDependencyConsumer DepConsumer(WorkerBundleHandle->SeenModules);

    if (WorkerBundleHandle->Worker->computeDependencies(
            /*WorkingDirectory*/ ".", CC1CommandLine, DepConsumer,
            LookupController, DiagConsumer))
      MaybeTUDeps = DepConsumer.takeTranslationUnitDeps();
  }

  return {std::move(MaybeTUDeps), DiagConsumer.takeDiagnostics()};
}

namespace {
struct DependencyScanResult {
  /// Indices of jobs that were successfully scanned.
  SmallVector<size_t> ScannedJobIndices;

  /// Input dependencies for scanned jobs. Parallel to \c ScannedJobIndices.
  SmallVector<InputDependencies, 0> InputDepsForScannedJobs;

  /// Module dependency graphs for scanned jobs. Parallel to \c
  /// ScannedJobIndices.
  SmallVector<deps::ModuleDepsGraph, 0> ModuleDepGraphsForScannedJobs;

  /// Indices of Standard library module jobs not discovered as dependencies.
  SmallVector<size_t> UnusedStdlibModuleJobIndices;

  /// Indices of jobs that could not be scanned (e.g. image jobs, ...).
  SmallVector<size_t> NonScannableJobIndices;
};
} // anonymous namespace

/// Scans the compilations job list \p Jobs for module dependencies.
///
/// Standard library module jobs are scanned on demand if imported by any
/// user-provided input.
///
/// \returns DependencyScanResult on success, or std::nullopt on failure, with
/// diagnostics reported via \p Diags in both cases.
static std::optional<DependencyScanResult> scanDependencies(
    ArrayRef<std::unique_ptr<Command>> Jobs,
    llvm::DenseMap<StringRef, const StdModuleManifest::Module *> ManifestLookup,
    StringRef ModuleCachePath, IntrusiveRefCntPtr<llvm::vfs::FileSystem> BaseFS,
    DiagnosticsEngine &Diags) {
  llvm::PrettyStackTraceString CrashInfo("Performing module dependency scan.");

  // Classify the jobs based on scan eligibility.
  SmallVector<size_t> ScannableJobIndices;
  SmallVector<size_t> NonScannableJobIndices;
  for (auto &&[Index, Job] : llvm::enumerate(Jobs)) {
    if (isDependencyScannableJob(*Job))
      ScannableJobIndices.push_back(Index);
    else
      NonScannableJobIndices.push_back(Index);
  }

  // Classify scannable jobs by origin. User-provided inputs will be scanned
  // immediately, while Standard library modules are indexed for on-demand
  // scanning when discovered as dependencies.
  SmallVector<size_t> UserInputScanIndices;
  llvm::DenseMap<ModuleNameAndTriple, size_t> StdlibModuleScanIndexByID;
  for (auto &&[ScanIndex, JobIndex] : llvm::enumerate(ScannableJobIndices)) {
    const Command &ScanJob = *Jobs[JobIndex];
    if (const auto *Entry =
            getManifestEntryForCommand(ScanJob, ManifestLookup)) {
      ModuleNameAndTriple ID{Entry->LogicalName, getTriple(ScanJob)};
      const bool Inserted =
          StdlibModuleScanIndexByID.try_emplace(ID, ScanIndex).second;
      assert(Inserted &&
             "Multiple jobs build the same module for the same triple.");
    } else {
      UserInputScanIndices.push_back(ScanIndex);
    }
  }

  // Initialize the scan context.
  const size_t NumScanInputs = ScannableJobIndices.size();
  const bool HasStdlibModuleInputs = !StdlibModuleScanIndexByID.empty();

  deps::DependencyScanningServiceOptions Opts;
  Opts.MakeVFS = [&] { return BaseFS; };
  deps::DependencyScanningService ScanningService(std::move(Opts));

  std::unique_ptr<llvm::ThreadPoolInterface> ThreadPool;
  std::unique_ptr<ScanningWorkerPool> WorkerPool;
  std::tie(ThreadPool, WorkerPool) = createOptimalThreadAndWorkerPool(
      NumScanInputs, HasStdlibModuleInputs, ScanningService);

  StdlibModuleScanScheduler StdlibModuleRegistry(StdlibModuleScanIndexByID);
  ModuleLookupController LookupController(ModuleCachePath);

  // Scan results are indexed by ScanIndex into ScannableJobIndices, not by
  // JobIndex into Jobs. This allows one result slot per scannable job.
  SmallVector<std::optional<deps::TranslationUnitDeps>, 0> AllScanResults(
      NumScanInputs);
  SmallVector<SmallVector<StandaloneDiagnostic, 0>, 0> AllScanDiags(
      NumScanInputs);
  std::atomic<bool> HasError{false};

  // Scans the job at the given scan index and schedules scans for any newly
  // discovered Standard library module dependencies.
  std::function<void(size_t)> ScanOneAndScheduleNew;
  ScanOneAndScheduleNew = [&](size_t ScanIndex) {
    const size_t JobIndex = ScannableJobIndices[ScanIndex];
    const Command &Job = *Jobs[JobIndex];
    auto [MaybeTUDeps, ScanDiags] =
        scanDependenciesForJob(Job, *WorkerPool, LookupController);

    // Store diagnostics even for successful scans to also capture any warnings
    // or notes.
    assert(AllScanDiags[ScanIndex].empty() &&
           "Each slot should be written to at most once.");
    AllScanDiags[ScanIndex] = std::move(ScanDiags);

    if (!MaybeTUDeps) {
      HasError.store(true, std::memory_order_relaxed);
      return;
    }

    // Schedule scans for newly discovered Standard library module dependencies.
    const auto NewScanInputs = StdlibModuleRegistry.getNewScanInputs(
        MaybeTUDeps->NamedModuleDeps, getTriple(Job));
    for (const size_t NewScanIndex : NewScanInputs)
      ThreadPool->async(
          [&, NewScanIndex]() { ScanOneAndScheduleNew(NewScanIndex); });

    assert(!AllScanResults[ScanIndex].has_value() &&
           "Each slot should be written to at most once.");
    AllScanResults[ScanIndex] = std::move(MaybeTUDeps);
  };

  // Initiate the scan with all jobs for user-provided inputs.
  for (const size_t ScanIndex : UserInputScanIndices)
    ThreadPool->async([&ScanOneAndScheduleNew, ScanIndex]() {
      ScanOneAndScheduleNew(ScanIndex);
    });
  ThreadPool->wait();

  reportAllScanDiagnostics(std::move(AllScanDiags), Diags);
  if (HasError.load(std::memory_order_relaxed))
    return std::nullopt;

  // Collect results, mapping scan indices back to job indices.
  DependencyScanResult Result;
  for (auto &&[JobIndex, MaybeTUDeps] :
       llvm::zip_equal(ScannableJobIndices, AllScanResults)) {
    if (MaybeTUDeps) {
      Result.ScannedJobIndices.push_back(JobIndex);
      Result.ModuleDepGraphsForScannedJobs.push_back(
          std::move(MaybeTUDeps->ModuleGraph));
      Result.InputDepsForScannedJobs.push_back(
          makeInputDeps(std::move(*MaybeTUDeps)));
    } else
      Result.UnusedStdlibModuleJobIndices.push_back(JobIndex);
  }
  Result.NonScannableJobIndices = std::move(NonScannableJobIndices);

#ifndef NDEBUG
  llvm::SmallDenseSet<size_t> SeenJobIndices;
  SeenJobIndices.insert_range(Result.ScannedJobIndices);
  SeenJobIndices.insert_range(Result.UnusedStdlibModuleJobIndices);
  SeenJobIndices.insert_range(Result.NonScannableJobIndices);
  assert(llvm::all_of(llvm::index_range(0, Jobs.size()),
                      [&](size_t JobIndex) {
                        return SeenJobIndices.contains(JobIndex);
                      }) &&
         "Scan result must partition all jobs");
#endif

  return Result;
}

namespace {
class CGNode;
class CGEdge;
using CGNodeBase = llvm::DGNode<CGNode, CGEdge>;
using CGEdgeBase = llvm::DGEdge<CGNode, CGEdge>;
using CGBase = llvm::DirectedGraph<CGNode, CGEdge>;

/// Compilation Graph Node
class CGNode : public CGNodeBase {
public:
  enum class NodeKind {
    ClangModuleCC1Job,
    NamedModuleCC1Job,
    NonModuleCC1Job,
    MiscJob,
    ImageJob,
    Root,
  };

  CGNode(const NodeKind K) : Kind(K) {}
  CGNode(const CGNode &) = delete;
  CGNode(CGNode &&) = delete;
  virtual ~CGNode() = 0;

  NodeKind getKind() const { return Kind; }

private:
  NodeKind Kind;
};
CGNode::~CGNode() = default;

/// Subclass of CGNode representing the root node of the graph.
///
/// The root node is a special node that connects to all other nodes with
/// no incoming edges, so that there is always a path from it to any node
/// in the graph.
///
/// There should only be one such node in a given graph.
class RootNode : public CGNode {
public:
  RootNode() : CGNode(NodeKind::Root) {}
  ~RootNode() override = default;

  static bool classof(const CGNode *N) {
    return N->getKind() == NodeKind::Root;
  }
};

/// Base class for any CGNode type that represents a job.
class JobNode : public CGNode {
public:
  JobNode(std::unique_ptr<Command> &&Job, NodeKind Kind)
      : CGNode(Kind), Job(std::move(Job)) {}
  virtual ~JobNode() override = 0;

  std::unique_ptr<Command> Job;

  static bool classof(const CGNode *N) {
    return N->getKind() != NodeKind::Root;
  }
};
JobNode::~JobNode() = default;

/// Subclass of CGNode representing a -cc1 job which produces a Clang module.
class ClangModuleJobNode : public JobNode {
public:
  ClangModuleJobNode(std::unique_ptr<Command> &&Job, deps::ModuleDeps &&MD)
      : JobNode(std::move(Job), NodeKind::ClangModuleCC1Job),
        MD(std::move(MD)) {}
  ~ClangModuleJobNode() override = default;

  deps::ModuleDeps MD;

  static bool classof(const CGNode *N) {
    return N->getKind() == NodeKind::ClangModuleCC1Job;
  }
};

/// Base class for any CGNode type that represents any scanned -cc1 job.
class ScannedJobNode : public JobNode {
public:
  ScannedJobNode(std::unique_ptr<Command> &&Job, InputDependencies &&InputDeps,
                 NodeKind Kind)
      : JobNode(std::move(Job), Kind), InputDeps(std::move(InputDeps)) {}
  ~ScannedJobNode() override = default;

  InputDependencies InputDeps;

  static bool classof(const CGNode *N) {
    return N->getKind() == NodeKind::NamedModuleCC1Job ||
           N->getKind() == NodeKind::NonModuleCC1Job;
  }
};

/// Subclass of CGNode representing a -cc1 job which produces a C++20 named
/// module.
class NamedModuleJobNode : public ScannedJobNode {
public:
  NamedModuleJobNode(std::unique_ptr<Command> &&Job,
                     InputDependencies &&InputDeps)
      : ScannedJobNode(std::move(Job), std::move(InputDeps),
                       NodeKind::NamedModuleCC1Job) {}
  ~NamedModuleJobNode() override = default;

  static bool classof(const CGNode *N) {
    return N->getKind() == NodeKind::NamedModuleCC1Job;
  }
};

/// Subclass of CGNode representing a -cc1 job which does not produce any
/// module, but might still have module imports.
class NonModuleTUJobNode : public ScannedJobNode {
public:
  NonModuleTUJobNode(std::unique_ptr<Command> &&Job,
                     InputDependencies &&InputDeps)
      : ScannedJobNode(std::move(Job), std::move(InputDeps),
                       NodeKind::NonModuleCC1Job) {}
  ~NonModuleTUJobNode() override = default;

  static bool classof(const CGNode *N) {
    return N->getKind() == NodeKind::NonModuleCC1Job;
  }
};

/// Subclass of CGNode representing a job which produces an image file, such as
/// a linker or interface stub merge job.
class ImageJobNode : public JobNode {
public:
  ImageJobNode(std::unique_ptr<Command> &&Job)
      : JobNode(std::move(Job), NodeKind::ImageJob) {}
  ~ImageJobNode() override = default;

  static bool classof(const CGNode *N) {
    return N->getKind() == NodeKind::ImageJob;
  }
};

/// Subclass of CGNode representing any job not covered by the other node types.
///
/// Jobs represented by this node type are not modified by the modules driver.
class MiscJobNode : public JobNode {
public:
  MiscJobNode(std::unique_ptr<Command> &&Job)
      : JobNode(std::move(Job), NodeKind::MiscJob) {}
  ~MiscJobNode() override = default;

  static bool classof(const CGNode *N) {
    return N->getKind() == NodeKind::MiscJob;
  }
};

/// Compilation Graph Edge
///
/// Edges connect the producer of an output to its consumer, except for edges
/// stemming from the root node.
class CGEdge : public CGEdgeBase {
public:
  enum class EdgeKind {
    Regular,
    ModuleDependency,
    Rooted,
  };

  CGEdge(CGNode &N, EdgeKind K) : CGEdgeBase(N), Kind(K) {}

  EdgeKind getKind() const { return Kind; }

private:
  EdgeKind Kind;
};

/// Compilation Graph
///
/// The graph owns all of its components.
/// All nodes and edges created by the graph have the same livetime as the
/// graph, even if removed from the graph's node list.
class CompilationGraph : public CGBase {
public:
  CompilationGraph() = default;
  CompilationGraph(const CompilationGraph &) = delete;
  CompilationGraph(CompilationGraph &&G) = default;

  CGNode &getRoot() const {
    assert(Root && "Root node has not yet been created!");
    return *Root;
  }

  RootNode &createRoot() {
    assert(!Root && "Root node has already been created!");
    auto &RootRef = createNodeImpl<RootNode>();
    Root = &RootRef;
    return RootRef;
  }

  template <typename T, typename... Args> T &createJobNode(Args &&...Arg) {
    static_assert(std::is_base_of<JobNode, T>::value,
                  "T must be derived from JobNode");
    return createNodeImpl<T>(std::forward<Args>(Arg)...);
  }

  CGEdge &createEdge(CGEdge::EdgeKind Kind, CGNode &Src, CGNode &Dst) {
    auto Edge = std::make_unique<CGEdge>(Dst, Kind);
    CGEdge &EdgeRef = *Edge;
    AllEdges.push_back(std::move(Edge));
    connect(Src, Dst, EdgeRef);
    return EdgeRef;
  }

private:
  using CGBase::addNode;
  using CGBase::connect;

  template <typename T, typename... Args> T &createNodeImpl(Args &&...Arg) {
    auto Node = std::make_unique<T>(std::forward<Args>(Arg)...);
    T &NodeRef = *Node;
    AllNodes.push_back(std::move(Node));
    addNode(NodeRef);
    return NodeRef;
  }

  CGNode *Root = nullptr;
  SmallVector<std::unique_ptr<CGNode>> AllNodes;
  SmallVector<std::unique_ptr<CGEdge>> AllEdges;
};
} // anonymous namespace

static StringRef getFirstInputFilename(const Command &Job) {
  return Job.getInputInfos().front().getFilename();
}

namespace llvm {
/// Non-const versions of the GraphTraits specializations for CompilationGraph.
template <> struct GraphTraits<CGNode *> {
  using NodeRef = CGNode *;

  static NodeRef CGGetTargetNode(CGEdge *E) { return &E->getTargetNode(); }

  using ChildIteratorType =
      mapped_iterator<CGNode::iterator, decltype(&CGGetTargetNode)>;
  using ChildEdgeIteratorType = CGNode::iterator;

  static NodeRef getEntryNode(NodeRef N) { return N; }

  static ChildIteratorType child_begin(NodeRef N) {
    return ChildIteratorType(N->begin(), &CGGetTargetNode);
  }

  static ChildIteratorType child_end(NodeRef N) {
    return ChildIteratorType(N->end(), &CGGetTargetNode);
  }

  static ChildEdgeIteratorType child_edge_begin(NodeRef N) {
    return N->begin();
  }
  static ChildEdgeIteratorType child_edge_end(NodeRef N) { return N->end(); }
};

template <> struct GraphTraits<CompilationGraph *> : GraphTraits<CGNode *> {
  using GraphRef = CompilationGraph *;
  using NodeRef = CGNode *;

  using nodes_iterator = CompilationGraph::iterator;

  static NodeRef getEntryNode(GraphRef G) { return &G->getRoot(); }

  static nodes_iterator nodes_begin(GraphRef G) { return G->begin(); }

  static nodes_iterator nodes_end(GraphRef G) { return G->end(); }
};

/// Const versions of the GraphTraits specializations for CompilationGraph.
template <> struct GraphTraits<const CGNode *> {
  using NodeRef = const CGNode *;

  static NodeRef CGGetTargetNode(const CGEdge *E) {
    return &E->getTargetNode();
  }

  using ChildIteratorType =
      mapped_iterator<CGNode::const_iterator, decltype(&CGGetTargetNode)>;
  using ChildEdgeIteratorType = CGNode::const_iterator;

  static NodeRef getEntryNode(NodeRef N) { return N; }

  static ChildIteratorType child_begin(NodeRef N) {
    return ChildIteratorType(N->begin(), &CGGetTargetNode);
  }

  static ChildIteratorType child_end(NodeRef N) {
    return ChildIteratorType(N->end(), &CGGetTargetNode);
  }

  static ChildEdgeIteratorType child_edge_begin(NodeRef N) {
    return N->begin();
  }

  static ChildEdgeIteratorType child_edge_end(NodeRef N) { return N->end(); }
};

template <>
struct GraphTraits<const CompilationGraph *> : GraphTraits<const CGNode *> {
  using GraphRef = const CompilationGraph *;
  using NodeRef = const CGNode *;

  using nodes_iterator = CompilationGraph::const_iterator;

  static NodeRef getEntryNode(GraphRef G) { return &G->getRoot(); }

  static nodes_iterator nodes_begin(GraphRef G) { return G->begin(); }

  static nodes_iterator nodes_end(GraphRef G) { return G->end(); }
};

template <>
struct DOTGraphTraits<const CompilationGraph *> : DefaultDOTGraphTraits {
  explicit DOTGraphTraits(bool IsSimple = false)
      : DefaultDOTGraphTraits(IsSimple) {}
  using GraphRef = const CompilationGraph *;
  using NodeRef = const CGNode *;

  static std::string getGraphName(GraphRef) {
    return "Module Dependency Graph";
  }

  static std::string getGraphProperties(GraphRef) {
    return "\tnode [shape=Mrecord, colorscheme=set23, style=filled];\n";
  }

  static bool renderGraphFromBottomUp() { return true; }

  static bool isNodeHidden(NodeRef N, GraphRef) {
    // Only show nodes with module dependency relations.
    return !isa<ClangModuleJobNode, ScannedJobNode>(N);
  }

  static std::string getNodeIdentifier(NodeRef N, GraphRef) {
    return llvm::TypeSwitch<NodeRef, std::string>(N)
        .Case([](const ClangModuleJobNode *ClangModuleNode) {
          const auto &ID = ClangModuleNode->MD.ID;
          return llvm::formatv("{0}-{1}", ID.ModuleName, ID.ContextHash).str();
        })
        .Case([](const NamedModuleJobNode *NamedModuleNode) {
          return llvm::formatv("{0}-{1}", NamedModuleNode->InputDeps.ModuleName,
                               getTriple(*NamedModuleNode->Job))
              .str();
        })
        .Case([](const NonModuleTUJobNode *NonModTUNode) {
          const auto &Job = *NonModTUNode->Job;
          return llvm::formatv("{0}-{1}", getFirstInputFilename(Job),
                               getTriple(Job))
              .str();
        })
        .DefaultUnreachable("Unexpected node kind! Is this node hidden?");
  }

  static std::string getNodeLabel(NodeRef N, GraphRef) {
    return llvm::TypeSwitch<NodeRef, std::string>(N)
        .Case([](const ClangModuleJobNode *ClangModuleNode) {
          const auto &ID = ClangModuleNode->MD.ID;
          return llvm::formatv("Module type: Clang module \\| Module name: {0} "
                               "\\| Hash: {1}",
                               ID.ModuleName, ID.ContextHash)
              .str();
        })
        .Case([](const NamedModuleJobNode *NamedModuleNode) {
          const auto &Job = *NamedModuleNode->Job;
          return llvm::formatv(
                     "Filename: {0} \\| Module type: Named module \\| "
                     "Module name: {1} \\| Triple: {2}",
                     getFirstInputFilename(Job),
                     NamedModuleNode->InputDeps.ModuleName, getTriple(Job))
              .str();
        })
        .Case([](const NonModuleTUJobNode *NonModTUNode) {
          const auto &Job = *NonModTUNode->Job;
          return llvm::formatv("Filename: {0} \\| Triple: {1}",
                               getFirstInputFilename(Job), getTriple(Job))
              .str();
        })
        .DefaultUnreachable("Unexpected node kind! Is this node hidden?");
  }

  static std::string getNodeAttributes(NodeRef N, GraphRef) {
    switch (N->getKind()) {
    case CGNode::NodeKind::ClangModuleCC1Job:
      return "fillcolor=1";
    case CGNode::NodeKind::NamedModuleCC1Job:
      return "fillcolor=2";
    case CGNode::NodeKind::NonModuleCC1Job:
      return "fillcolor=3";
    default:
      llvm_unreachable("Unexpected node kind! Is this node hidden?");
    }
  }
};

/// GraphWriter specialization for CompilationGraph that emits a more
/// human-readable DOT graph.
template <>
class GraphWriter<const CompilationGraph *>
    : public GraphWriterBase<const CompilationGraph *,
                             GraphWriter<const CompilationGraph *>> {
public:
  using GraphType = const CompilationGraph *;
  using Base = GraphWriterBase<GraphType, GraphWriter<GraphType>>;

  GraphWriter(llvm::raw_ostream &O, const GraphType &G, bool IsSimple)
      : Base(O, G, IsSimple), EscapedIDByNodeRef(G->size()) {}

  void writeNodes() {
    auto IsNodeVisible = [&](NodeRef N) { return !DTraits.isNodeHidden(N, G); };
    auto VisibleNodes = llvm::filter_to_vector(nodes(G), IsNodeVisible);

    writeNodeDefinitions(VisibleNodes);
    O << "\n";
    writeNodeRelations(VisibleNodes);
  }

private:
  using Base::DOTTraits;
  using Base::GTraits;
  using Base::NodeRef;

  void writeNodeDefinitions(ArrayRef<NodeRef> VisibleNodes) {
    for (NodeRef Node : VisibleNodes) {
      std::string EscapedNodeID =
          DOT::EscapeString(DTraits.getNodeIdentifier(Node, G));
      const std::string NodeLabel = DTraits.getNodeLabel(Node, G);
      const std::string NodeAttrs = DTraits.getNodeAttributes(Node, G);
      O << '\t' << '"' << EscapedNodeID << "\" [" << NodeAttrs << ", label=\"{ "
        << DOT::EscapeString(NodeLabel) << " }\"];\n";
      EscapedIDByNodeRef.try_emplace(Node, std::move(EscapedNodeID));
    }
  }

  void writeNodeRelations(ArrayRef<NodeRef> VisibleNodes) {
    auto IsNodeVisible = [&](NodeRef N) { return !DTraits.isNodeHidden(N, G); };
    for (NodeRef Node : VisibleNodes) {
      auto TgtNodes = llvm::make_range(GTraits::child_begin(Node),
                                       GTraits::child_end(Node));
      auto VisibleTgtNodes = llvm::make_filter_range(TgtNodes, IsNodeVisible);
      StringRef EscapedSrcNodeID = EscapedIDByNodeRef.at(Node);
      for (NodeRef TgtNode : VisibleTgtNodes) {
        StringRef EscapedTgtNodeID = EscapedIDByNodeRef.at(TgtNode);
        O << '\t' << '"' << EscapedSrcNodeID << "\" -> \"" << EscapedTgtNodeID
          << "\";\n";
      }
    }
  }

  DenseMap<NodeRef, std::string> EscapedIDByNodeRef;
};
} // namespace llvm

static SmallVector<std::unique_ptr<Command>>
takeJobsAtIndices(SmallVectorImpl<std::unique_ptr<Command>> &Jobs,
                  ArrayRef<size_t> Indices) {
  SmallVector<std::unique_ptr<Command>> Out;
  for (const auto JobIndex : Indices) {
    assert(Jobs[JobIndex] && "Expected valid job!");
    Out.push_back(std::move(Jobs[JobIndex]));
  }
  return Out;
}

/// Creates nodes for all jobs that could not be scanned (e.g. image jobs, ...).
static void createNodesForNonScannableJobs(
    CompilationGraph &Graph,
    SmallVectorImpl<std::unique_ptr<Command>> &&NonScannableJobs) {
  for (auto &Job : NonScannableJobs) {
    if (Job->getCreator().isLinkJob())
      Graph.createJobNode<ImageJobNode>(std::move(Job));
    else
      Graph.createJobNode<MiscJobNode>(std::move(Job));
  }
}

/// Creates nodes for the Standard library module jobs not discovered as
/// dependencies.
///
/// These and any dependent (non-image) nodes  be removed later.
static SmallVector<CGNode *> createNodesForUnusedStdlibModuleJobs(
    CompilationGraph &Graph,
    SmallVectorImpl<std::unique_ptr<Command>> &&UnusedStdlibModuleJobs) {
  SmallVector<CGNode *> StdlibModuleNodesToPrune;
  for (auto &Job : UnusedStdlibModuleJobs) {
    auto &NewNode = Graph.createJobNode<MiscJobNode>(std::move(Job));
    StdlibModuleNodesToPrune.push_back(&NewNode);
  }
  return StdlibModuleNodesToPrune;
}

/// Creates a job for the Clang module described by \p MD.
static std::unique_ptr<CC1Command>
createClangModulePrecompileJob(Compilation &C, const Command &ImportingJob,
                               const deps::ModuleDeps &MD) {
  DerivedArgList &Args = C.getArgs();
  const OptTable &Opts = C.getDriver().getOpts();
  Arg *InputArg = makeInputArg(Args, Opts, "<discovered clang module>");
  Action *IA = C.MakeAction<InputAction>(*InputArg, types::ID::TY_ModuleFile);
  Action *PA = C.MakeAction<PrecompileJobAction>(IA, types::ID::TY_ModuleFile);
  PA->propagateOffloadInfo(&ImportingJob.getSource());

  auto &TC = ImportingJob.getCreator().getToolChain();
  auto &TCArgs = C.getArgsForToolChain(&TC, PA->getOffloadingArch(),
                                       PA->getOffloadingDeviceKind());

  auto BuildArgs = MD.getBuildArguments();
  ArgStringList JobArgs;
  JobArgs.reserve(BuildArgs.size());
  for (const auto &Arg : BuildArgs)
    JobArgs.push_back(TCArgs.MakeArgString(Arg));

  return std::make_unique<CC1Command>(
      *PA, ImportingJob.getCreator(), ResponseFileSupport::AtFileUTF8(),
      C.getDriver().getClangProgramPath(), JobArgs,
      /*Inputs=*/ArrayRef<InputInfo>{},
      /*Outputs=*/ArrayRef<InputInfo>{});
}

/// Creates a ClangModuleJobNode and its job for each unique Clang module
/// in \p ModuleDepGraphsForScannedJobs.
///
/// Only the jobs at indices \p ScannedJobIndices in \p Jobs are expected to be
/// non-null.
static void createClangModuleJobsAndNodes(
    CompilationGraph &Graph, Compilation &C,
    ArrayRef<std::unique_ptr<Command>> Jobs, ArrayRef<size_t> ScannedJobIndices,
    SmallVectorImpl<deps::ModuleDepsGraph> &&ModuleDepGraphsForScannedJobs) {
  llvm::DenseSet<deps::ModuleID> AlreadySeen;
  for (auto &&[ScanIndex, ModuleDepsGraph] :
       llvm::enumerate(ModuleDepGraphsForScannedJobs)) {
    const auto &ImportingJob = *Jobs[ScannedJobIndices[ScanIndex]];

    for (auto &MD : ModuleDepsGraph) {
      const auto Inserted = AlreadySeen.insert(MD.ID).second;
      if (!Inserted)
        continue;

      auto ClangModuleJob = createClangModulePrecompileJob(C, ImportingJob, MD);
      Graph.createJobNode<ClangModuleJobNode>(std::move(ClangModuleJob),
                                              std::move(MD));
    }
  }
}

/// Creates nodes for all jobs which were scanned for dependencies.
///
/// The updated command lines produced by the dependency scan are installed at a
/// later point.
static void createNodesForScannedJobs(
    CompilationGraph &Graph,
    SmallVectorImpl<std::unique_ptr<Command>> &&ScannedJobs,
    SmallVectorImpl<InputDependencies> &&InputDepsForScannedJobs) {
  for (auto &&[Job, InputDeps] :
       llvm::zip_equal(ScannedJobs, InputDepsForScannedJobs)) {
    if (InputDeps.ModuleName.empty())
      Graph.createJobNode<NonModuleTUJobNode>(std::move(Job),
                                              std::move(InputDeps));
    else
      Graph.createJobNode<NamedModuleJobNode>(std::move(Job),
                                              std::move(InputDeps));
  }
}

template <typename LookupT, typename KeyRangeT>
static void connectEdgesViaLookup(CompilationGraph &Graph, CGNode &TgtNode,
                                  const LookupT &SrcNodeLookup,
                                  const KeyRangeT &SrcNodeLookupKeys,
                                  CGEdge::EdgeKind Kind) {
  for (const auto &Key : SrcNodeLookupKeys) {
    const auto It = SrcNodeLookup.find(Key);
    if (It == SrcNodeLookup.end())
      continue;

    auto &SrcNode = *It->second;
    Graph.createEdge(Kind, SrcNode, TgtNode);
  }
}

/// Create edges for regular (non-module) dependencies in \p Graph.
static void createRegularEdges(CompilationGraph &Graph) {
  llvm::DenseMap<StringRef, CGNode *> NodeByOutputFiles;
  for (auto &Node : Graph) {
    for (const auto &Output : cast<JobNode>(Node)->Job->getOutputFilenames()) {
      const bool Inserted = NodeByOutputFiles.try_emplace(Output, Node).second;
      assert(Inserted &&
             "Driver should not produce multiple jobs with identical outputs!");
    }
  }

  for (auto &Node : Graph) {
    const auto &InputInfos = cast<JobNode>(Node)->Job->getInputInfos();
    auto InputFilenames = llvm::map_range(
        InputInfos, [](const auto &II) { return II.getFilename(); });

    connectEdgesViaLookup(Graph, *Node, NodeByOutputFiles, InputFilenames,
                          CGEdge::EdgeKind::Regular);
  }
}

/// Create edges for regular (non-module) dependencies in \p Graph.
static bool createModuleDependencyEdges(CompilationGraph &Graph,
                                        DiagnosticsEngine &Diags) {
  llvm::DenseMap<deps::ModuleID, CGNode *> ClangModuleNodeByID;
  llvm::DenseMap<ModuleNameAndTriple, CGNode *> NamedModuleNodeByID;

  // Map each module to the job that produces it.
  bool HasDuplicateModuleError = false;
  for (auto &Node : Graph) {
    llvm::TypeSwitch<CGNode *>(Node)
        .Case([&](ClangModuleJobNode *ClangModuleNode) {
          const bool Inserted =
              ClangModuleNodeByID.try_emplace(ClangModuleNode->MD.ID, Node)
                  .second;
          assert(Inserted &&
                 "Multiple Clang module nodes with the same module ID!");
        })
        .Case([&](NamedModuleJobNode *NamedModuleNode) {
          StringRef ModuleName = NamedModuleNode->InputDeps.ModuleName;
          ModuleNameAndTriple ID{ModuleName, getTriple(*NamedModuleNode->Job)};
          const auto [It, Inserted] = NamedModuleNodeByID.try_emplace(ID, Node);
          if (!Inserted) {
            // For scan input jobs, their first input is always a filename and
            // the scanned source.
            // We don't use InputDeps.FileDeps here because diagnostics should
            // refer to the filename as specified on the command line, not the
            // canonical absolute path.
            StringRef PrevFile =
                getFirstInputFilename(*cast<JobNode>(It->second)->Job);
            StringRef CurFile = getFirstInputFilename(*NamedModuleNode->Job);
            Diags.Report(diag::err_modules_driver_named_module_redefinition)
                << ModuleName << PrevFile << CurFile;
            HasDuplicateModuleError = true;
          }
        });
  }
  if (HasDuplicateModuleError)
    return false;

  // Create edges from the module nodes to their importers.
  for (auto &Node : Graph) {
    llvm::TypeSwitch<CGNode *>(Node)
        .Case([&](ClangModuleJobNode *ClangModuleNode) {
          connectEdgesViaLookup(Graph, *ClangModuleNode, ClangModuleNodeByID,
                                ClangModuleNode->MD.ClangModuleDeps,
                                CGEdge::EdgeKind::ModuleDependency);
        })
        .Case([&](ScannedJobNode *NodeWithInputDeps) {
          connectEdgesViaLookup(Graph, *NodeWithInputDeps, ClangModuleNodeByID,
                                NodeWithInputDeps->InputDeps.ClangModuleDeps,
                                CGEdge::EdgeKind::ModuleDependency);

          StringRef Triple = getTriple(*NodeWithInputDeps->Job);
          const auto NamedModuleDepIDs =
              llvm::map_range(NodeWithInputDeps->InputDeps.NamedModuleDeps,
                              [&](StringRef ModuleName) {
                                return ModuleNameAndTriple{ModuleName, Triple};
                              });
          connectEdgesViaLookup(Graph, *NodeWithInputDeps, NamedModuleNodeByID,
                                NamedModuleDepIDs,
                                CGEdge::EdgeKind::ModuleDependency);
        });
  }

  return true;
}

/// Prunes the compilation graph of any jobs which build Standard library
/// modules not required in this compilation.
static void
pruneUnusedStdlibModuleJobs(CompilationGraph &Graph,
                            ArrayRef<CGNode *> UnusedStdlibModuleNodes) {
  // Collect all reachable nodes holding non-image jobs.
  llvm::SmallPtrSet<CGNode *, 16> DeadNodes;
  for (auto *UnusedStdlibModuleNode : UnusedStdlibModuleNodes) {
    auto ReachableNonImageNodes = llvm::make_filter_range(
        llvm::depth_first(cast<CGNode>(UnusedStdlibModuleNode)),
        std::not_fn(llvm::IsaPred<ImageJobNode>));

    DeadNodes.insert_range(ReachableNonImageNodes);
  }

  // Map image nodes to the dead nodes that feed into them.
  llvm::DenseMap<ImageJobNode *, llvm::SmallPtrSet<CGNode *, 4>>
      DeadJobsByImageJob;
  for (auto *DeadNode : DeadNodes) {
    auto ReachableImageNodes = llvm::make_filter_range(
        llvm::depth_first(DeadNode), llvm::IsaPred<ImageJobNode>);
    for (auto *ImageNode : ReachableImageNodes)
      DeadJobsByImageJob[cast<ImageJobNode>(ImageNode)].insert(DeadNode);
  }

  // Remove from each affected image node any arguments corresponding to
  // outputs of dead nodes.
  for (auto &[ImageNode, DeadJobNodes] : DeadJobsByImageJob) {
    SmallVector<StringRef, 4> OutputsToRemove;
    for (auto *DeadNode : DeadJobNodes)
      llvm::append_range(OutputsToRemove,
                         cast<JobNode>(DeadNode)->Job->getOutputFilenames());

    auto NewArgs = ImageNode->Job->getArguments();
    llvm::erase_if(NewArgs, [&](StringRef Arg) {
      return llvm::is_contained(OutputsToRemove, Arg);
    });
    ImageNode->Job->replaceArguments(NewArgs);
  }

  // Erase all dead nodes from the graph.
  for (auto *DeadNode : DeadNodes) {
    // Nodes are owned by the graph, but we can release the associated job.
    cast<JobNode>(DeadNode)->Job.reset();
    Graph.removeNode(*DeadNode);
  }
}

/// Creates the root node and connects it to all nodes with no incoming edges
/// ensuring that every node in the graph is reachable from the root.
static void createAndConnectRoot(CompilationGraph &Graph) {
  llvm::SmallPtrSet<CGNode *, 16> HasIncomingEdge;
  for (auto *Node : Graph)
    for (auto *Edge : Node->getEdges())
      HasIncomingEdge.insert(&Edge->getTargetNode());

  auto AllNonRootNodes = llvm::iterator_range(Graph);
  auto &Root = Graph.createRoot();

  for (auto *Node : AllNonRootNodes) {
    if (HasIncomingEdge.contains(Node))
      continue;
    Graph.createEdge(CGEdge::EdgeKind::Rooted, Root, *Node);
  }
}

void driver::modules::runModulesDriver(
    Compilation &C, ArrayRef<StdModuleManifest::Module> ManifestEntries) {
  llvm::PrettyStackTraceString CrashInfo("Running modules driver.");
  DiagnosticsEngine &Diags = C.getDriver().getDiags();

  const auto MaybeModuleCachePath = getModuleCachePath(C.getArgs());
  if (!MaybeModuleCachePath) {
    Diags.Report(diag::err_default_modules_cache_not_available);
    return;
  }

  auto Jobs = C.getJobs().takeJobs();

  // Apply manifest-specified local arguments before scanning as they may affect
  // the scan results.
  const auto ManifestEntryBySource = buildManifestLookupMap(ManifestEntries);
  applyLocalArgsFromManifest(C, ManifestEntryBySource, Jobs);

  // Run the dependency scan.
  auto MaybeScanResults =
      scanDependencies(Jobs, ManifestEntryBySource, *MaybeModuleCachePath,
                       &C.getDriver().getVFS(), Diags);
  if (!MaybeScanResults) {
    Diags.Report(diag::err_dependency_scan_failed);
    return;
  }
  auto &ScanResult = *MaybeScanResults;

  // Build the compilation graph.
  CompilationGraph Graph;
  createNodesForNonScannableJobs(
      Graph, takeJobsAtIndices(Jobs, ScanResult.NonScannableJobIndices));
  auto UnusedStdlibModuleNodes = createNodesForUnusedStdlibModuleJobs(
      Graph, takeJobsAtIndices(Jobs, ScanResult.UnusedStdlibModuleJobIndices));
  createClangModuleJobsAndNodes(
      Graph, C, Jobs, ScanResult.ScannedJobIndices,
      std::move(ScanResult.ModuleDepGraphsForScannedJobs));
  createNodesForScannedJobs(
      Graph, takeJobsAtIndices(Jobs, ScanResult.ScannedJobIndices),
      std::move(ScanResult.InputDepsForScannedJobs));
  createRegularEdges(Graph);

  pruneUnusedStdlibModuleJobs(Graph, UnusedStdlibModuleNodes);

  if (!createModuleDependencyEdges(Graph, Diags))
    return;
  createAndConnectRoot(Graph);

  Diags.Report(diag::remark_printing_module_graph);
  if (!Diags.isLastDiagnosticIgnored())
    llvm::WriteGraph<const CompilationGraph *>(llvm::errs(), &Graph);

  // TODO: Install all updated command-lines produced by the dependency scan.
  // TODO: Fix-up command-lines for named module imports.
  // TODO: Sort the graph topologically and feed jobs back into the Compilation.
}
