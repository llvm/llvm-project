//===- DependencyScanner.cpp - Module dependency discovery ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Driver/DependencyScanner.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/InputInfo.h"
#include "clang/Driver/Job.h"
#include "clang/Driver/Options.h"
#include "clang/Driver/Tool.h"
#include "clang/Lex/Lexer.h"
#include "clang/Tooling/DependencyScanning/DependencyScanningTool.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/DOTGraphTraits.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/TargetParser/Host.h"

using namespace clang::tooling::dependencies;
using namespace clang;
using namespace llvm::opt;

//===----------------------------------------------------------------------===//
// Dependency Scan Diagnostic Reporting Utilities
//===----------------------------------------------------------------------===//

namespace {
/// Represents a CharSourceRange within a StandaloneDiagnostic.
struct SourceOffsetRange {
  SourceOffsetRange(CharSourceRange Range, const SourceManager &SrcMgr,
                    const LangOptions &LangOpts);
  unsigned Begin = 0;
  unsigned End = 0;
  bool IsTokenRange = false;
};

/// Represents a FixItHint within a StandaloneDiagnostic.
struct StandaloneFixIt {
  StandaloneFixIt(const SourceManager &SrcMgr, const LangOptions &LangOpts,
                  const FixItHint &FixIt);

  SourceOffsetRange RemoveRange;
  SourceOffsetRange InsertFromRange;
  std::string CodeToInsert;
  bool BeforePreviousInsertions = false;
};

/// Represents a StoredDiagnostic in a form that can be retained until after its
/// SourceManager has been destroyed.
///
/// Source locations are stored as a combination of filename and offsets into
/// that file.
/// To report the diagnostic, it must first be translated back into a
/// StoredDiagnostic with a new associated SourceManager.
struct StandaloneDiagnostic {
  explicit StandaloneDiagnostic(const StoredDiagnostic &StoredDiag);

  LangOptions LangOpts;
  SrcMgr::CharacteristicKind FileKind;
  DiagnosticsEngine::Level Level;
  unsigned ID = 0;
  unsigned FileOffset = 0;
  std::string Filename;
  std::string Message;
  SmallVector<SourceOffsetRange, 0> Ranges;
  SmallVector<StandaloneFixIt, 0> FixIts;
};
} // anonymous namespace

SourceOffsetRange::SourceOffsetRange(CharSourceRange Range,
                                     const SourceManager &SrcMgr,
                                     const LangOptions &LangOpts)
    : IsTokenRange(Range.isTokenRange()) {
  const auto FileRange = Lexer::makeFileCharRange(Range, SrcMgr, LangOpts);
  Begin = SrcMgr.getFileOffset(FileRange.getBegin());
  End = SrcMgr.getFileOffset(FileRange.getEnd());
}

StandaloneFixIt::StandaloneFixIt(const SourceManager &SrcMgr,
                                 const LangOptions &LangOpts,
                                 const FixItHint &FixIt)
    : RemoveRange(FixIt.RemoveRange, SrcMgr, LangOpts),
      InsertFromRange(FixIt.InsertFromRange, SrcMgr, LangOpts),
      CodeToInsert(FixIt.CodeToInsert),
      BeforePreviousInsertions(FixIt.BeforePreviousInsertions) {}

/// If a custom working directory is set for \c SrcMgr, returns the absolute
/// path of \c Filename to make it independent. Otherwise, returns the original
/// string.
static std::string canonicalizeFilename(const SourceManager &SrcMgr,
                                        StringRef Filename) {
  SmallString<256> Abs(Filename);
  if (!llvm::sys::path::is_absolute(Abs)) {
    if (const auto &CWD =
            SrcMgr.getFileManager().getFileSystemOpts().WorkingDir;
        !CWD.empty())
      llvm::sys::fs::make_absolute(CWD, Abs);
  }
  return std::string(Abs.str());
}

// FIXME: LangOpts is not properly saved because the LangOptions is not
// copyable!
StandaloneDiagnostic::StandaloneDiagnostic(const StoredDiagnostic &StoredDiag)
    : Level(StoredDiag.getLevel()), ID(StoredDiag.getID()),
      Message(StoredDiag.getMessage()) {
  const FullSourceLoc &FullLoc = StoredDiag.getLocation();
  // This is not an invalid diagnostic; invalid SourceLocations are used to
  // represent diagnostics without a specific SourceLocation.
  if (FullLoc.isInvalid())
    return;

  const auto &SrcMgr = FullLoc.getManager();
  FileKind = SrcMgr.getFileCharacteristic(static_cast<SourceLocation>(FullLoc));
  const auto FileLoc = SrcMgr.getFileLoc(static_cast<SourceLocation>(FullLoc));
  FileOffset = SrcMgr.getFileOffset(FileLoc);
  const auto PathRef = SrcMgr.getFilename(FileLoc);
  assert(!PathRef.empty() && "diagnostic with location has no source file?");
  Filename = canonicalizeFilename(SrcMgr, PathRef);

  Ranges.reserve(StoredDiag.getRanges().size());
  for (const auto &Range : StoredDiag.getRanges())
    Ranges.emplace_back(Range, SrcMgr, LangOpts);

  FixIts.reserve(StoredDiag.getFixIts().size());
  for (const auto &FixIt : StoredDiag.getFixIts())
    FixIts.emplace_back(SrcMgr, LangOpts, FixIt);
}

/// Translates \c StandaloneDiag into a StoredDiagnostic, associating it with
/// the provided FileManager and SourceManager.
static StoredDiagnostic
translateStandaloneDiag(FileManager &FileMgr, SourceManager &SrcMgr,
                        StandaloneDiagnostic &&StandaloneDiag) {
  const auto FileRef = FileMgr.getOptionalFileRef(StandaloneDiag.Filename);
  if (!FileRef)
    return StoredDiagnostic(StandaloneDiag.Level, StandaloneDiag.ID,
                            std::move(StandaloneDiag.Message));

  const auto FileID =
      SrcMgr.getOrCreateFileID(*FileRef, StandaloneDiag.FileKind);
  const auto FileLoc = SrcMgr.getLocForStartOfFile(FileID);
  assert(FileLoc.isValid() && "StandaloneDiagnostic should only use FilePath "
                              "for encoding a valid source location.");
  const auto DiagLoc = FileLoc.getLocWithOffset(StandaloneDiag.FileOffset);
  const FullSourceLoc Loc(DiagLoc, SrcMgr);

  auto ConvertOffsetRange = [&](const SourceOffsetRange &Range) {
    return CharSourceRange(SourceRange(FileLoc.getLocWithOffset(Range.Begin),
                                       FileLoc.getLocWithOffset(Range.End)),
                           Range.IsTokenRange);
  };
  SmallVector<CharSourceRange> TranslatedRanges;
  TranslatedRanges.reserve(StandaloneDiag.Ranges.size());
  llvm::transform(StandaloneDiag.Ranges, std::back_inserter(TranslatedRanges),
                  ConvertOffsetRange);

  SmallVector<FixItHint> TranslatedFixIts;
  TranslatedFixIts.reserve(StandaloneDiag.FixIts.size());
  for (const auto &FixIt : StandaloneDiag.FixIts) {
    FixItHint TranslatedFixIt;
    TranslatedFixIt.CodeToInsert = std::string(FixIt.CodeToInsert);
    TranslatedFixIt.RemoveRange = ConvertOffsetRange(FixIt.RemoveRange);
    TranslatedFixIt.InsertFromRange = ConvertOffsetRange(FixIt.InsertFromRange);
    TranslatedFixIt.BeforePreviousInsertions = FixIt.BeforePreviousInsertions;
    TranslatedFixIts.push_back(std::move(TranslatedFixIt));
  }

  return StoredDiagnostic(StandaloneDiag.Level, StandaloneDiag.ID,
                          StandaloneDiag.Message, Loc, TranslatedRanges,
                          TranslatedFixIts);
}

namespace {
/// RAII utility to report StandaloneDiagnostics through a DiagnosticsEngine.
///
/// The driver's DiagnosticsEngine usually does not have a SourceManager at this
/// point in building the compilation, in which case the
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

  void Report(StandaloneDiagnostic &&StandaloneDiag) const {
    const auto StoredDiag = translateStandaloneDiag(
        getFileManager(), getSourceManager(), std::move(StandaloneDiag));
    Diags.getClient()->BeginSourceFile(StandaloneDiag.LangOpts, nullptr);
    Diags.Report(StoredDiag);
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

namespace {
/// Collects diagnostics in a form that can be retained until after their
/// associated SourceManager is destroyed.
class StandaloneDiagCollector : public DiagnosticConsumer {
public:
  void BeginSourceFile(const LangOptions &LangOpts,
                       const Preprocessor *PP = nullptr) override {}

  void HandleDiagnostic(DiagnosticsEngine::Level Level,
                        const Diagnostic &Info) override {
    StoredDiagnostic StoredDiag(Level, Info);
    StandaloneDiags.emplace_back(StoredDiag);
  }

  void EndSourceFile() override {}

  SmallVector<StandaloneDiagnostic, 0> takeDiags() {
    SmallVector<StandaloneDiagnostic, 0> Out;
    Out.swap(StandaloneDiags);
    return Out;
  }

private:
  SmallVector<StandaloneDiagnostic, 0> StandaloneDiags;
};
} // anonymous namespace

//===----------------------------------------------------------------------===//
// Dependency Scan: Scanning Task Generation
//===----------------------------------------------------------------------===//

namespace {
/// Represents a dependency scanning task for a single source input.
struct CC1ScanTask {
  CC1ScanTask(size_t JobIndex, std::string &&Filename,
              std::vector<std::string> &&CC1CommandLine)
      : JobIndex(JobIndex), Filename(std::move(Filename)),
        CC1CommandLine(std::move(CC1CommandLine)) {}

  /// Index of the corresponding job inside the replicated driver from which
  /// this scanning task was generated. This index is used to correlate the
  /// produced scan result with the job that the calling driver will later
  /// create.
  size_t JobIndex;

  /// The source input for which this task was generated.
  std::string Filename;

  /// The -cc1 command line to invoke the dependency scan with.
  std::vector<std::string> CC1CommandLine;
};
} // anonymous namespace

/// Non-consuming version of llvm::opt::ArgList::AddAllArgsExcept.
///
/// Appends all arguments onto \c Output as strings, except those which match
/// any option defined in \c ExcludedIds.
static void addAllArgsExcept(const DerivedArgList &Args,
                             ArrayRef<OptSpecifier> ExcludedIds,
                             ArgStringList &Output) {
  using namespace llvm::opt;
  auto IsNotExcluded = [&](const Arg *Arg) -> bool {
    return llvm::none_of(ExcludedIds, [&](const OptSpecifier Id) {
      return Arg->getOption().matches(Id);
    });
  };
  for (const auto *Arg : llvm::make_filter_range(Args, IsNotExcluded))
    Arg->render(Args, Output);
}

static ArgStringList buildDummyDriverCommandLine(StringRef ClangProgramPath,
                                                 const DerivedArgList &Args) {
  using namespace driver::options;
  const SmallVector<OptSpecifier, 1> ExcludedIds{OPT_ccc_print_phases};
  ArgStringList CommandLine{ClangProgramPath.data()};
  addAllArgsExcept(Args, ExcludedIds, CommandLine);
  CommandLine.push_back("-fno-modules-driver");
  return CommandLine;
}

/// Returns true if a driver job is a viable dependency scan input.
static bool isJobForDepScan(const driver::Command &Job) {
  if (StringRef(Job.getCreator().getName()) != "clang")
    return false;
  auto IsSrcInput = [](const driver::InputInfo &II) -> bool {
    return isSrcFile(II.getType());
  };
  return llvm::all_of(Job.getInputInfos(), IsSrcInput);
}

/// Given a -cc1 driver job, generates its full command line.
static std::vector<std::string>
buildCC1CommandLineFromJob(const driver::Command &Job,
                           DiagnosticsEngine &Diags) {
  CompilerInvocation CI;
  CompilerInvocation::CreateFromArgs(CI, Job.getArguments(), Diags);
  std::vector<std::string> CL{"clang", "-cc1"};
  CI.generateCC1CommandLine([&](const Twine &Arg) { CL.push_back(Arg.str()); });
  return CL;
}

/// Generates the list of dependency scanning tasks from the given driver
/// command line.
///
/// The calling driver has not constructed its compilation jobs yet, but we
/// require the resulting -cc1 command lines for this compilation as inputs
/// to the dependency scan.
/// To obtain them, we create a replica of the calling driver and let it
/// construct the full compilation. We then extract -cc1 jobs from that
/// compilation.
static SmallVector<CC1ScanTask, 0>
buildScanTasksFromDummyDriver(StringRef ClangExecutable,
                              DiagnosticsEngine &Diags,
                              const DerivedArgList &Args) {
  SmallVector<CC1ScanTask, 0> Out;

  const auto DummyDriver = std::make_unique<driver::Driver>(
      ClangExecutable, llvm::sys::getDefaultTargetTriple(), Diags);
  // This has already been handled by the calling driver.
  DummyDriver->setCheckInputsExist(false);
  DummyDriver->setTitle("Driver for dependency scan -cc1 input generation");
  const auto DummyDriverCL = buildDummyDriverCommandLine(ClangExecutable, Args);
  std::unique_ptr<driver::Compilation> C(
      DummyDriver->BuildCompilation(DummyDriverCL));
  if (!C)
    return Out;
  if (C->containsError()) {
    llvm::reportFatalInternalError(
        "failed to construct the compilation required to generate -cc1 input "
        "command lines for the dependency scan");
  }

  for (const auto &[Index, Job] : llvm::enumerate(C->getJobs())) {
    if (!isJobForDepScan(Job))
      continue;
    // Heuristic: pick the first source input as the primary input.
    std::string Filename;
    if (!Job.getInputInfos().empty())
      Filename = Job.getInputInfos().front().getFilename();
    auto CC1Command = buildCC1CommandLineFromJob(Job, Diags);
    Out.emplace_back(Index, std::move(Filename), std::move(CC1Command));
  }

  return Out;
}

//===----------------------------------------------------------------------===//
// Dependency Scan
//===----------------------------------------------------------------------===//

namespace clang {
namespace driver::dependencies {

namespace {
/// A simple dependency action controller that only provides module lookup for
/// Clang modules.
class ModuleLookupActionController : public DependencyActionController {
public:
  ModuleLookupActionController() {
    Driver::getDefaultModuleCachePath(ModulesCachePath);
  }

  std::string lookupModuleOutput(const ClangModuleDeps &MD,
                                 ModuleOutputKind Kind) override {
    if (Kind == ModuleOutputKind::ModuleFile)
      return constructPCMPath(MD.ID);
    // Driver command lines which cause this should be handled either in
    // Driver::handleArguments and rejected or in
    // buildCommandLineForDummyDriver and modified.
    llvm::reportFatalInternalError(
        "call to lookupModuleOutput with unexpected ModuleOutputKind");
  }

private:
  SmallString<128> ModulesCachePath;

  std::string constructPCMPath(const ModuleID &MID) const {
    SmallString<256> ExplicitPCMPath(ModulesCachePath);
    llvm::sys::path::append(ExplicitPCMPath, Twine(MID.ModuleName) + "-" +
                                                 MID.ContextHash + ".pcm");
    return std::string(ExplicitPCMPath);
  }
};

struct ScanResultInternal {
  /// On success, the full dependencies and Clang module graph for the input.
  std::optional<TranslationUnitDeps> MaybeTUDeps;

  /// Diagnostics omitted by the dependency scanning tool.
  SmallVector<StandaloneDiagnostic, 0> Diagnostics;
};
} // anonymous namespace

/// Runs the dependency scanning tool for every task using work-stealing
/// concurrency.
static SmallVector<ScanResultInternal, 0>
scanModuleDependenciesInternal(const ArrayRef<CC1ScanTask> Tasks) {
  SmallVector<ScanResultInternal, 0> Out(Tasks.size());

  const IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS =
      llvm::vfs::createPhysicalFileSystem();
  DependencyScanningService Service(ScanningMode::DependencyDirectivesScan,
                                    ScanningOutputFormat::Full);

  std::atomic<size_t> NextIndex = 0;
  auto GetNextTaskIndex = [&]() -> std::optional<size_t> {
    if (const auto CurIndex = NextIndex.fetch_add(1, std::memory_order_relaxed);
        CurIndex < Tasks.size())
      return CurIndex;
    return std::nullopt;
  };

  auto RunScanningWorker = [&]() -> void {
    DependencyScanningWorker Worker(Service, FS);
    llvm::DenseSet<ModuleID> AlreadySeenModules;
    constexpr StringRef CWD(".");

    while (auto MaybeIndex = GetNextTaskIndex()) {
      const size_t I = *MaybeIndex;
      const auto &CC1CommandLine = Tasks[I].CC1CommandLine;
      ModuleLookupActionController LookupController;
      StandaloneDiagCollector DiagConsumer;
      FullDependencyConsumer FullDepsConsumer(AlreadySeenModules);

      const bool Success =
          Worker.computeDependencies(CWD, CC1CommandLine, FullDepsConsumer,
                                     LookupController, DiagConsumer);
      Out[I].MaybeTUDeps =
          Success ? std::optional(FullDepsConsumer.takeTranslationUnitDeps())
                  : std::nullopt;
      Out[I].Diagnostics = DiagConsumer.takeDiags();
    }
  };

  if (Tasks.size() <= 1)
    RunScanningWorker();
  else {
    llvm::DefaultThreadPool Pool;
    // Avoid launching more threads than inputs.
    const size_t Concurrency =
        std::min(static_cast<size_t>(Pool.getMaxConcurrency()), Tasks.size());
    for (size_t I = 0; I < Concurrency; ++I)
      Pool.async(RunScanningWorker);
    Pool.wait();
  }

  return Out;
}

Expected<SmallVector<TranslationUnitScanResult, 0>>
scanDependencies(StringRef ClangExecutable, DiagnosticsEngine &Diags,
                 const DerivedArgList &Args) {
  auto ScanTasks = buildScanTasksFromDummyDriver(ClangExecutable, Diags, Args);
  auto InternalScanResultList = scanModuleDependenciesInternal(ScanTasks);
  assert(ScanTasks.size() == InternalScanResultList.size() &&
         "expected a scan result for every scan task");

  SmallVector<TranslationUnitScanResult, 0> Out;
  Out.reserve(InternalScanResultList.size());

  const StandaloneDiagReporter DiagReporter(Diags);
  bool HasFailure = false;

  for (size_t I = 0, E = ScanTasks.size(); I < E; ++I) {
    auto &Item = ScanTasks[I];
    auto &[MaybeTUDeps, Diagnostics] = InternalScanResultList[I];

    for (auto &StandaloneDiag : Diagnostics)
      DiagReporter.Report(std::move(StandaloneDiag));

    if (!MaybeTUDeps) {
      HasFailure = true;
      Diags.Report(diag::remark_failed_dependency_scan_for_input)
          << Item.Filename;
      continue;
    }

    Out.emplace_back(Item.JobIndex, std::move(Item.Filename),
                     std::move(*MaybeTUDeps));
  }

  if (HasFailure)
    return llvm::createStringError("Failed to perform dependency scan");
  return Out;
}

//===----------------------------------------------------------------------===//
// Module Dependency Graph
//===----------------------------------------------------------------------===//

MDGNode::~MDGNode() = default;

namespace detail {
class ModuleDepGraphBuilder {
public:
  explicit ModuleDepGraphBuilder(
      SmallVectorImpl<TranslationUnitScanResult> &&ScanResults,
      DiagnosticsEngine &Diags)
      : Graph(std::move(ScanResults)),
        BackingScanResults(Graph.BackingScanResults),
        Allocator(Graph.BumpPtrAlloc), Diags(Diags) {}

  /// Builds the full module dependency graph.
  ///
  /// \returns false on error (with diagnostics reported to \c Diags).
  bool build();

  /// Takes ownership of the constructed dependency graph.
  ModuleDepGraph takeGraph() { return std::move(Graph); }

private:
  ModuleDepGraph Graph;
  const SmallVectorImpl<TranslationUnitScanResult> &BackingScanResults;
  llvm::BumpPtrAllocator &Allocator;

  DiagnosticsEngine &Diags;

  /// Allocation helper using the graph's allocator.
  template <typename MDGComponent, typename... Args>
  MDGComponent *makeWithGraphAlloc(Args &&...args) {
    return new (Allocator.Allocate(sizeof(MDGComponent), alignof(MDGComponent)))
        MDGComponent(std::forward<Args>(args)...);
  }

  /// Lookup maps used for connecting the nodes.
  llvm::DenseMap<ModuleID, MDGNode *> ClangModuleNodeMap;
  llvm::StringMap<NamedCXXModuleMDGNode *> CXXNamedModuleMap;

  bool buildNodes();
  void addClangModuleGraph(const ClangModuleGraph &ModuleGraph);
  void addClangModuleNode(const ClangModuleDeps &ModuleDeps);
  void addNonModuleNode(const TranslationUnitScanResult &ScanResult);
  bool addCXXNamedModuleNode(const TranslationUnitScanResult &ScanResult);

  void connectNodes();
  bool connectDepsForNode(MDGNode *Importer);
  void addImportEdge(MDGNode &Imported, MDGNode &Importer);
};

bool ModuleDepGraphBuilder::build() {
  if (!buildNodes())
    return false;
  connectNodes();
  return true;
}

/// Constructs every node in the graph from the collected scan results.
///
/// \returns false if an error occurs.
bool ModuleDepGraphBuilder::buildNodes() {
  Graph.Root = makeWithGraphAlloc<RootMDGNode>();
  for (const auto &ScanResult : BackingScanResults) {
    const auto &TUDeps = ScanResult.TUDeps;
    addClangModuleGraph(TUDeps.ModuleGraph);
    // If ModuleName is empty, this translation unit is not a module.
    if (!TUDeps.ID.ModuleName.empty()) {
      bool Success = addCXXNamedModuleNode(ScanResult);
      if (!Success)
        return false;
    } else {
      addNonModuleNode(ScanResult);
    }
  }
  return true;
}

/// Adds nodes for any entry in this Clang module graph that we have not seen
/// yet.
void ModuleDepGraphBuilder::addClangModuleGraph(
    const tooling::dependencies::ModuleDepsGraph &ClangModuleGraph) {
  for (const auto &ClangModuleDeps : ClangModuleGraph) {
    const ModuleID &ID = ClangModuleDeps.ID;
    // Skip if we have already added a node for this graph entry.
    if (ClangModuleNodeMap.contains(ID))
      continue;
    addClangModuleNode(ClangModuleDeps);
  }
}

/// Adds a node representing the Clang module unit described by \c
/// ClangModuleDeps.
void ModuleDepGraphBuilder::addClangModuleNode(
    const ClangModuleDeps &ClangModuleDeps) {
  auto *Node = makeWithGraphAlloc<ClangModuleMDGNode>(ClangModuleDeps);
  const auto [_, Inserted] =
      ClangModuleNodeMap.try_emplace(ClangModuleDeps.ID, Node);
  assert(Inserted && "Duplicate nodes for a single Clang module!");
  Graph.addNode(*Node);
}

/// Adds a node representing the C++ named module unit described by \c
/// ScanResult.
///
/// \returns false on error if a node for the same module interface already
/// exists in the graph.
bool ModuleDepGraphBuilder::addCXXNamedModuleNode(
    const TranslationUnitScanResult &ScanResult) {
  auto *Node = makeWithGraphAlloc<NamedCXXModuleMDGNode>(ScanResult);
  StringRef ModuleName = ScanResult.TUDeps.ID.ModuleName;

  const auto [It, Inserted] = CXXNamedModuleMap.try_emplace(ModuleName, Node);
  if (!Inserted) {
    StringRef ExistingFile = It->second->getFilename();
    StringRef ThisFile = ScanResult.Filename;
    Diags.Report(diag::err_mod_graph_named_module_redefinition)
        << ModuleName << ExistingFile << ThisFile;
    return false;
  }

  Graph.addNode(*Node);
  return true;
}

/// Adds a node representing the non-module translation unit described by \c
/// ScanResult.
void ModuleDepGraphBuilder::addNonModuleNode(
    const TranslationUnitScanResult &ScanResult) {
  auto *Node = makeWithGraphAlloc<NonModuleMDGNode>(ScanResult);
  Graph.addNode(*Node);
}

/// Wires dependency edges for every node in the graph.
///
/// Edges are directed from the imported module to the importing unit.
/// If a node has no incoming edges, connects it to the root.
void ModuleDepGraphBuilder::connectNodes() {
  for (auto *Node : nodes(&Graph)) {
    bool HasIncomingEdge = connectDepsForNode(Node);
    if (!HasIncomingEdge)
      addImportEdge(*Graph.getRoot(), *Node);
  }
}

/// Connects all dependencies for a node.
///
/// \returns true if any edge was added.
bool ModuleDepGraphBuilder::connectDepsForNode(MDGNode *Importer) {
  bool HasAnyImport = false;
  // Connect Clang module dependencies.
  for (const auto &MID : Importer->getClangModuleDeps()) {
    if (MDGNode *Imported = ClangModuleNodeMap.lookup(MID)) {
      addImportEdge(*Imported, *Importer);
      HasAnyImport = true;
    }
  }
  // Connect C++20 named module dependencies.
  for (const auto &Name : Importer->getCXXNamedModuleDeps()) {
    if (auto *Imported = CXXNamedModuleMap.lookup(Name)) {
      addImportEdge(*Imported, *Importer);
      HasAnyImport = true;
    }
  }
  return HasAnyImport;
}

/// Creates and adds an edge from \c Imported to \c Importer.
void ModuleDepGraphBuilder::addImportEdge(MDGNode &Imported,
                                          MDGNode &Importer) {
  auto *Edge = makeWithGraphAlloc<MDGEdge>(Importer);
  Imported.addEdge(*Edge);
}

} // namespace detail

llvm::Expected<ModuleDepGraph>
buildModuleDepGraph(SmallVectorImpl<TranslationUnitScanResult> &&Scans,
                    DiagnosticsEngine &Diags) {
  detail::ModuleDepGraphBuilder Builder(std::move(Scans), Diags);
  if (!Builder.build())
    return llvm::createStringError(
        "Failed to construct module dependency graph");
  return Builder.takeGraph();
}
} // namespace driver::dependencies
} // namespace clang

//===----------------------------------------------------------------------===//
// Module Dependency Graph: GraphViz Output
//===----------------------------------------------------------------------===//

namespace deps = clang::driver::dependencies;

namespace llvm {
template <>
struct DOTGraphTraits<const deps::ModuleDepGraph *> : DefaultDOTGraphTraits {
  explicit DOTGraphTraits(bool IsSimple = false)
      : DefaultDOTGraphTraits(IsSimple) {}

  static std::string getGraphName(const deps::ModuleDepGraph *) {
    return "Module Dependency Graph";
  }

  static std::string getGraphProperties(const deps::ModuleDepGraph *) {
    return "\tnode [shape=Mrecord];\n";
  }

  static bool isNodeHidden(const deps::MDGNode *N,
                           const deps::ModuleDepGraph *) {
    return isa<deps::RootMDGNode>(N);
  }

  static std::string getNodeIdentifier(const deps::MDGNode *N,
                                       const deps::ModuleDepGraph *) {
    llvm::SmallString<128> Buf;
    llvm::raw_svector_ostream OS(Buf);
    llvm::TypeSwitch<const deps::MDGNode *>(N)
        .Case<deps::ClangModuleMDGNode, deps::NamedCXXModuleMDGNode>(
            [&](auto *N) { OS << N->getModuleID().ModuleName; })
        .Case<deps::NonModuleMDGNode>([&](auto *N) { OS << N->getFilename(); })
        .Default([](auto *) { llvm_unreachable("Unhandled MDGNode kind!"); });
    OS << " (" << getNodeKindStr(N->getKind()) << ")";

    return std::string(OS.str());
  }

  static std::string getNodeLabel(const deps::MDGNode *N,
                                  const deps::ModuleDepGraph *) {
    SmallString<128> Buf;
    raw_svector_ostream OS(Buf);
    OS << "Type: " << getNodeKindStr(N->getKind()) << " \\| ";

    auto PrintFilename = [](raw_ostream &OS, StringRef Filename) {
      OS << "Filename: " << Filename;
    };
    auto PrintModuleName = [](raw_ostream &OS, StringRef ModuleName) {
      OS << "Provides: " << ModuleName;
    };
    llvm::TypeSwitch<const deps::MDGNode *>(N)
        .Case<const deps::ClangModuleMDGNode>(
            [&](auto *N) { PrintModuleName(OS, N->getModuleID().ModuleName); })
        .Case<const deps::NamedCXXModuleMDGNode>([&](auto *N) {
          PrintModuleName(OS, N->getModuleID().ModuleName);
          OS << " \\| ";
          PrintFilename(OS, N->getFilename());
        })
        .Case<const deps::NonModuleMDGNode>(
            [&](auto *N) { PrintFilename(OS, N->getFilename()); })
        .Default([](auto *) {
          llvm::reportFatalInternalError("Unhandled MDGNode kind!");
        });

    return std::string(OS.str());
  }

private:
  static StringRef getNodeKindStr(deps::MDGNode::NodeKind Kind) {
    using NodeKind = deps::MDGNode::NodeKind;
    switch (Kind) {
    case NodeKind::NamedCXXModule:
      return "C++20 module";
    case NodeKind::ClangModule:
      return "Clang module";
    case NodeKind::NonModule:
      return "Non-module source";
    default:
      llvm::reportFatalInternalError("Unexpected MDGNode kind!");
    };
  }
};

/// GraphWriter specialization for ModuleDepGraph.
///
/// Uses human-readable identifiers instead of raw pointers and separates node
/// definitions from import edges for a more remark-friendly output.
template <>
class GraphWriter<const deps::ModuleDepGraph *>
    : public GraphWriterBase<const deps::ModuleDepGraph *,
                             GraphWriter<const deps::ModuleDepGraph *>> {
public:
  using GraphType = const deps::ModuleDepGraph *;
  using Base = GraphWriterBase<GraphType, GraphWriter<GraphType>>;

  GraphWriter(llvm::raw_ostream &O, const GraphType &G, bool IsSimple)
      : Base(O, G, IsSimple) {}

  void writeNodes();

private:
  using Base::DOTTraits;
  using Base::GTraits;
  using Base::NodeRef;

  DenseMap<NodeRef, std::string> NodeIDMap;

  void writeNodeDefinitions(ArrayRef<NodeRef> Nodes);
  void writeNodeRelations(ArrayRef<NodeRef> Nodes);
};

void GraphWriter<const deps::ModuleDepGraph *>::writeNodes() {
  auto IsNodeVisible = [&](NodeRef N) { return !DTraits.isNodeHidden(N, G); };
  const auto VisibleNodeRange =
      llvm::make_filter_range(nodes(G), IsNodeVisible);
  const SmallVector<NodeRef, 0> VisibleNodes(VisibleNodeRange);

  writeNodeDefinitions(VisibleNodes);
  writeNodeRelations(VisibleNodes);
}

void GraphWriter<const deps::ModuleDepGraph *>::writeNodeDefinitions(
    ArrayRef<NodeRef> Nodes) {
  for (const auto &Node : Nodes) {
    const auto NodeID = DTraits.getNodeIdentifier(Node, G);
    const auto NodeLabel = DTraits.getNodeLabel(Node, G);
    O << "\t\"" << DOT::EscapeString(NodeID) << "\" [ "
      << DTraits.getNodeAttributes(Node, G) << "label=\"{ "
      << DOT::EscapeString(NodeLabel) << " }\"];\n";
    NodeIDMap.try_emplace(Node, std::move(NodeID));
  }
  O << "\n";
}

void GraphWriter<const deps::ModuleDepGraph *>::writeNodeRelations(
    ArrayRef<NodeRef> Nodes) {
  for (const auto &Node : Nodes) {
    const auto &SourceNodeID = NodeIDMap.at(Node);
    for (const auto &Edge : Node->getEdges()) {
      const auto *TargetNode = GTraits::MDGGetTargetNode(Edge);
      const auto &TargetNodeID = NodeIDMap.at(TargetNode);
      O << "\t\"" << DOT::EscapeString(SourceNodeID) << "\" -> \""
        << DOT::EscapeString(TargetNodeID) << "\";\n";
    }
  }
}
} // namespace llvm

namespace clang {
namespace driver::dependencies {
void writeModuleDepGraph(llvm::raw_ostream &OS, const ModuleDepGraph &G) {
  llvm::WriteGraph(OS, &G);
}
} // namespace driver::dependencies
} // namespace clang
