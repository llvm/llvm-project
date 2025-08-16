#include "clang/Driver/DependencyScanner.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Tool.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Tooling/DependencyScanning/DependencyScanningWorker.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/TargetParser/Host.h"
#include <atomic>

using namespace clang;
using namespace llvm::opt;

//===----------------------------------------------------------------------===//
// Dependency Scan Diagnostic Reporting Utilities
//===----------------------------------------------------------------------===//

namespace {
/// Utility class to represent a CharSourceRange in a StandaloneDiagnostic.
struct CharSourceOffsetRange {
  CharSourceOffsetRange(CharSourceRange Range, const SourceManager &SrcMgr,
                        const LangOptions &LangOpts);

  unsigned Begin = 0;
  unsigned End = 0;
  bool IsTokenRange = false;
};

/// Utility class to represent a FixItHint in a StandaloneDiagnostic.
struct StandaloneFixIt {
  StandaloneFixIt(const SourceManager &SrcMgr, const LangOptions &LangOpts,
                  const FixItHint &FixIt);

  CharSourceOffsetRange RemoveRange;
  CharSourceOffsetRange InsertFromRange;
  std::string CodeToInsert;
  bool BeforePreviousInsertions = false;
};

/// Represents a diagnostic in a form that can be retained until after its
/// corresponding source manager is destroyed.
///
/// Source locations are stored as offsets into the file FilePath.
/// To report a StandaloneDiagnostic, it must first be converted back into a
/// StoredDiagnostic with a new source manager.
struct StandaloneDiagnostic {
  StandaloneDiagnostic(const StoredDiagnostic &StoredDiag);

  LangOptions LangOpts;
  SrcMgr::CharacteristicKind FileCharacteristic;
  DiagnosticsEngine::Level Level;
  unsigned ID = 0;
  unsigned FileOffset = 0;
  std::string FilePath;
  std::string Message;
  SmallVector<CharSourceOffsetRange> Ranges;
  SmallVector<StandaloneFixIt> FixIts;
};
} // anonymous namespace

CharSourceOffsetRange::CharSourceOffsetRange(CharSourceRange Range,
                                             const SourceManager &SrcMgr,
                                             const LangOptions &LangOpts)
    : IsTokenRange(Range.isTokenRange()) {
  const auto FileRange =
      Lexer::makeFileCharRange(std::move(Range), SrcMgr, LangOpts);
  Begin = SrcMgr.getFileOffset(FileRange.getBegin());
  End = SrcMgr.getFileOffset(FileRange.getEnd());
}

StandaloneFixIt::StandaloneFixIt(const SourceManager &SrcMgr,
                                 const LangOptions &LangOpts,
                                 const FixItHint &FixIt)
    : RemoveRange(FixIt.RemoveRange, SrcMgr, LangOpts),
      InsertFromRange(FixIt.InsertFromRange, SrcMgr, LangOpts),
      CodeToInsert(std::move(FixIt.CodeToInsert)),
      BeforePreviousInsertions(FixIt.BeforePreviousInsertions) {}

StandaloneDiagnostic::StandaloneDiagnostic(const StoredDiagnostic &StoredDiag)
    : Level(StoredDiag.getLevel()), ID(StoredDiag.getID()),
      Message(StoredDiag.getMessage()), Ranges(), FixIts() {
  // This is not an invalid diagnostic; invalid SourceLocations are used to
  // represent diagnostics without a specific SourceLocation.
  if (StoredDiag.getLocation().isInvalid())
    return;

  const auto &SrcMgr = StoredDiag.getLocation().getManager();
  FileCharacteristic = SrcMgr.getFileCharacteristic(StoredDiag.getLocation());

  const auto FileLoc = SrcMgr.getFileLoc(StoredDiag.getLocation());
  StringRef Filename = SrcMgr.getFilename(FileLoc);
  assert(!Filename.empty() && "diagnostic with location has no source file?");
  // If a custom CWD is set by the FileManager, make FilePath independent by
  // converting it to an absolute path.
  SmallString<256> AbsFilePath(Filename);
  if (const auto &CWD = SrcMgr.getFileManager().getFileSystemOpts().WorkingDir;
      !CWD.empty())
    llvm::sys::fs::make_absolute(CWD, AbsFilePath);
  FilePath = AbsFilePath.str();
  FileOffset = SrcMgr.getFileOffset(FileLoc);

  Ranges.reserve(StoredDiag.getRanges().size());
  for (const auto &Range : StoredDiag.getRanges())
    Ranges.emplace_back(Range, SrcMgr, LangOpts);
  FixIts.reserve(StoredDiag.getFixIts().size());
  for (const auto &FixIt : StoredDiag.getFixIts())
    FixIts.emplace_back(SrcMgr, LangOpts, FixIt);
}

/// Converts a StandaloneDiagnostic into a StoredDiagnostic, associating it
/// with the provided FileManager and SourceManager.
static StoredDiagnostic
convertStandaloneDiagnostic(FileManager &FileMgr, SourceManager &SrcMgr,
                            StandaloneDiagnostic StandaloneDiag) {
  const auto FileRef = FileMgr.getOptionalFileRef(StandaloneDiag.FilePath);
  if (!FileRef)
    return StoredDiagnostic(StandaloneDiag.Level, StandaloneDiag.ID,
                            StandaloneDiag.Message);

  const auto FileID =
      SrcMgr.getOrCreateFileID(*FileRef, StandaloneDiag.FileCharacteristic);
  const auto FileLoc = SrcMgr.getLocForStartOfFile(FileID);
  assert(FileLoc.isValid() && "StandaloneDiagnostic should only use FilePath "
                              "for encoding a valid source location.");
  const auto DiagLoc = FileLoc.getLocWithOffset(StandaloneDiag.FileOffset);
  const FullSourceLoc Loc(DiagLoc, SrcMgr);

  auto ConvertOffsetRange = [&](const CharSourceOffsetRange &Range) {
    return CharSourceRange(SourceRange(FileLoc.getLocWithOffset(Range.Begin),
                                       FileLoc.getLocWithOffset(Range.End)),
                           Range.IsTokenRange);
  };

  SmallVector<clang::CharSourceRange> TranslatedRanges;
  TranslatedRanges.reserve(StandaloneDiag.Ranges.size());
  llvm::transform(StandaloneDiag.Ranges, TranslatedRanges.begin(),
                  ConvertOffsetRange);

  SmallVector<FixItHint> TranslatedFixIts;
  TranslatedFixIts.reserve(StandaloneDiag.FixIts.size());
  for (const auto &FixIt : StandaloneDiag.FixIts) {
    FixItHint TranslatedFixIt;
    TranslatedFixIt.CodeToInsert = std::string(FixIt.CodeToInsert);
    TranslatedFixIt.RemoveRange = ConvertOffsetRange(FixIt.RemoveRange);
    TranslatedFixIt.InsertFromRange = ConvertOffsetRange(FixIt.InsertFromRange);
    TranslatedFixIts.push_back(TranslatedFixIt);
  }

  return StoredDiagnostic(StandaloneDiag.Level, StandaloneDiag.ID,
                          StandaloneDiag.Message, Loc, TranslatedRanges,
                          TranslatedFixIts);
}

namespace {
/// Simple RAII helper to report StandaloneDiagnostics.
class StandaloneDiagReporter {
public:
  StandaloneDiagReporter(DiagnosticsEngine &Diags,
                         llvm::raw_ostream &OS = llvm::errs());

  StandaloneDiagReporter(const StandaloneDiagReporter &) = delete;
  StandaloneDiagReporter &operator=(const StandaloneDiagReporter &) = delete;

  ~StandaloneDiagReporter();

  void Report(StandaloneDiagnostic StandaloneDiag);

private:
  DiagnosticsEngine &Diags;
  FileManager *FileMgr = nullptr;
  SourceManager *SrcMgr = nullptr;
  bool OwnsManagers = false;
};
} // anonymous namespace

StandaloneDiagReporter::StandaloneDiagReporter(DiagnosticsEngine &Diags,
                                               llvm::raw_ostream &OS)
    : Diags(Diags) {
  // If Diags already has a source manager, use that one instead.
  if (Diags.hasSourceManager()) {
    OwnsManagers = false;
    SrcMgr = &Diags.getSourceManager();
    FileMgr = &SrcMgr->getFileManager();
    assert(FileMgr != nullptr &&
           "DiagnosticEngine with invalid SourceManager!");
  } else {
    OwnsManagers = true;
    FileMgr = new FileManager({"."});
    SrcMgr = new SourceManager(Diags, *FileMgr);
  }
}

StandaloneDiagReporter::~StandaloneDiagReporter() {
  if (OwnsManagers) {
    delete SrcMgr;
    delete FileMgr;
  }
}

void StandaloneDiagReporter::Report(StandaloneDiagnostic StandaloneDiag) {
  const auto StoredDiag =
      convertStandaloneDiagnostic(*FileMgr, *SrcMgr, std::move(StandaloneDiag));
  Diags.getClient()->BeginSourceFile(StandaloneDiag.LangOpts, nullptr);
  Diags.Report(StoredDiag);
  Diags.getClient()->EndSourceFile();
}

/// Reports all StandaloneDiagnostic instances collected over  multiple scans.
static void reportAllScanDiagnostics(
    DiagnosticsEngine &Diags,
    ArrayRef<SmallVector<StandaloneDiagnostic, 0>> StandaloneDiagLists) {
  StandaloneDiagReporter StandaloneDiagReporter(Diags);
  for (const auto &StandaloneDiags : StandaloneDiagLists)
    for (const auto &SD : StandaloneDiags)
      StandaloneDiagReporter.Report(SD);
}

namespace {
/// Collects every Diagnostic in a form that can be retained until after its
/// corresponding source manager is destroyed.
/// FIXME: Store the original LangOpts in the StoredDiagnostic too.
class StandaloneDiagCollector : public DiagnosticConsumer {
public:
  void BeginSourceFile(const LangOptions &LangOpts,
                       const Preprocessor *PP = nullptr) override {}

  void EndSourceFile() override {}

  void HandleDiagnostic(DiagnosticsEngine::Level Level,
                        const Diagnostic &Info) override {
    StoredDiagnostic StoredDiag(Level, Info);
    StandaloneDiags.emplace_back(StoredDiag);
  }

  SmallVector<StandaloneDiagnostic, 0> takeDiags() {
    return std::move(StandaloneDiags);
  }

private:
  SmallVector<StandaloneDiagnostic, 0> StandaloneDiags;
};
} // anonymous namespace

//===----------------------------------------------------------------------===//
// Dependency Scan Clang `-cc1` Command Line Generation
//===----------------------------------------------------------------------===//

using CC1Command = std::vector<std::string>;

/// Non-consuming version of llvm::opt::ArgList::AddAllArgsExcept.
/// Append all arguments onto the \c Output as strings, execpt those which match
/// any option defined in \c ExcludedIds.
static void addAllArgsExcept(const DerivedArgList &Args,
                             ArrayRef<llvm::opt::OptSpecifier> ExcludedIds,
                             ArgStringList &Output) {
  using namespace llvm::opt;
  auto IsNotExcluded = [&](const Arg *Arg) -> bool {
    return llvm::none_of(ExcludedIds, [&](const OptSpecifier Id) {
      return Arg->getOption().matches(Id);
    });
  };
  for (const Arg *Arg : llvm::make_filter_range(Args, IsNotExcluded))
    Arg->render(Args, Output);
}

/// Checks if a driver is a -cc1 command and if all its inputs are source files.
static bool isJobForDepScan(const driver::Command &Cmd) {
  const bool IsCC1Cmd = StringRef(Cmd.getCreator().getName()) == "clang";
  auto IsSrcInput = [](const driver::InputInfo &II) -> bool {
    return isSrcFile(II.getType());
  };
  return IsCC1Cmd && llvm::all_of(Cmd.getInputInfos(), IsSrcInput);
}

/// Builds the command line to create the dummy driver from the calling drivers
/// command line.
static ArgStringList
buildCommandLineForDummyDriver(StringRef ClangProgramPath,
                               const DerivedArgList &Args) {
  using namespace driver::options;
  const llvm::SmallVector<llvm::opt::OptSpecifier, 1> ExcludedIds{
      OPT_ccc_print_phases};
  ArgStringList CommandLine{ClangProgramPath.data()};
  addAllArgsExcept(Args, ExcludedIds, CommandLine);
  CommandLine.push_back("-fno-modules-driver");
  return CommandLine;
}

/// Builds all -cc1 command line inputs for dependency scanning.
///
/// Since the calling driver has not yet created the compilation jobs,
/// we reconstruct the driver and fully build the compilation to extract
/// the -cc1 command lines as inputs for the dependency scan.
static SmallVector<CC1Command>
buildCommandLinesFromDummyDriver(StringRef ClangProgramPath,
                                 DiagnosticsEngine &Diags,
                                 const DerivedArgList &Args) {
  const auto DummyCommandLine =
      buildCommandLineForDummyDriver(ClangProgramPath, Args);

  const auto DummyDriver = std::make_unique<driver::Driver>(
      DummyCommandLine[0], llvm::sys::getDefaultTargetTriple(), Diags);
  DummyDriver->setCheckInputsExist(false);
  DummyDriver->setTitle("Driver for dependency scan '-cc1' input generation");
  std::unique_ptr<driver::Compilation> C(
      DummyDriver->BuildCompilation(DummyCommandLine));
  if (!C)
    return {};
  if (C->containsError()) {
    llvm::reportFatalInternalError(
        "failed to construct the compilation required to generate '-cc1' input "
        "command lines for the dependency scan");
  }

  SmallVector<CC1Command> CC1CommandLines;
  const auto JobsForDepScan =
      llvm::make_filter_range(C->getJobs(), isJobForDepScan);
  for (const driver::Command &Job : JobsForDepScan) {
    auto CI = std::make_unique<CompilerInvocation>();
    CompilerInvocation::CreateFromArgs(*CI, Job.getArguments(), Diags);
    std::vector<std::string> CommandLine{ClangProgramPath.str(), "-cc1"};
    CI->generateCC1CommandLine(
        [&CommandLine](const Twine &Arg) { CommandLine.push_back(Arg.str()); });
    CC1CommandLines.push_back(std::move(CommandLine));
  }
  return CC1CommandLines;
}

//===----------------------------------------------------------------------===//
// Dependency Scan
//===----------------------------------------------------------------------===//

namespace clang {
namespace driver {
namespace dependencies {

char DependencyScanError::ID = 0;

namespace {
using namespace tooling::dependencies;
/// A simple dependency action controller that only provides module lookup for
/// Clang modules.
class ModuleLookupActionController : public DependencyActionController {
public:
  ModuleLookupActionController() {
    driver::Driver::getDefaultModuleCachePath(ModulesCachePath);
  }

  std::string lookupModuleOutput(const ModuleDeps &MD,
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

  std::string constructPCMPath(const ModuleID MID) {
    SmallString<256> ExplicitPCMPath(ModulesCachePath);
    llvm::sys::path::append(ExplicitPCMPath, llvm::Twine(MID.ContextHash) +
                                                 MID.ModuleName + "-" +
                                                 MID.ContextHash + ".pcm");
    return std::string(ExplicitPCMPath);
  }
};
} // anonymous namespace

/// Builds -cc1 command lines from the driver arguments, then performs a
/// dependency scan for each -cc1 command line using a work-stealing
/// concurrency.
llvm::Expected<SmallVector<TranslationUnitDeps, 0>>
scanModuleDependencies(llvm::StringRef ClangProgramPath,
                       DiagnosticsEngine &Diags,
                       const llvm::opt::DerivedArgList &Args) {
  llvm::PrettyStackTraceString CrashInfo("Scanning module dependencies");
  using namespace tooling::dependencies;

  const auto CC1CommandLines =
      buildCommandLinesFromDummyDriver(ClangProgramPath, Diags, Args);

  std::atomic<size_t> Index = 0;
  auto GetNextInputIndex = [&]() -> std::optional<size_t> {
    const auto CurIndex = Index.fetch_add(1, std::memory_order_relaxed);
    if (CurIndex < CC1CommandLines.size())
      return CurIndex;
    return std::nullopt;
  };

  DependencyScanningService Service(ScanningMode::DependencyDirectivesScan,
                                    ScanningOutputFormat::Full);
  llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS =
      llvm::vfs::createPhysicalFileSystem();

  SmallVector<TranslationUnitDeps, 0> ScanResults(CC1CommandLines.size());
  SmallVector<SmallVector<StandaloneDiagnostic, 0>> StandaloneDiagLists(
      CC1CommandLines.size());
  std::atomic<bool> HasScanningError = false;

  auto ScanningTask = [&]() -> void {
    DependencyScanningWorker Worker(Service, FS);
    llvm::DenseSet<ModuleID> AlreadySeenModules;
    constexpr auto CWD = ".";

    while (auto MaybeInputIndex = GetNextInputIndex()) {
      const auto &CommandLine = CC1CommandLines[*MaybeInputIndex];
      ModuleLookupActionController ModuleLookupController;
      StandaloneDiagCollector DiagsConsumer;
      FullDependencyConsumer FullDepsConsumer(AlreadySeenModules);

      const bool Success = Worker.computeDependencies(
          CWD, CommandLine, FullDepsConsumer, ModuleLookupController,
          DiagsConsumer, std::nullopt);
      if (!Success)
        HasScanningError.store(true, std::memory_order_relaxed);

      ScanResults[*MaybeInputIndex] =
          FullDepsConsumer.takeTranslationUnitDeps();
      StandaloneDiagLists[*MaybeInputIndex] = DiagsConsumer.takeDiags();
    }
  };

  if (CC1CommandLines.size() == 1)
    ScanningTask();
  else {
    llvm::DefaultThreadPool Pool;
    for (unsigned I = 0; I < Pool.getMaxConcurrency(); ++I)
      Pool.async(ScanningTask);
    Pool.wait();
  }

  reportAllScanDiagnostics(Diags, StandaloneDiagLists);
  if (HasScanningError)
    return llvm::make_error<DependencyScanError>();
  return ScanResults;
}

//===----------------------------------------------------------------------===//
// Module Dependency Graph
//===----------------------------------------------------------------------===//

MDGNode::~MDGNode() = default;

template <typename NodeTy, typename... Args>
NodeTy *ModuleDepGraph::MakeWithBumpAlloc(Args &&...args) {
  return new (BumpPtrAlloc.Allocate(sizeof(NodeTy), alignof(NodeTy)))
      NodeTy(std::forward<Args>(args)...);
}

/// Helper class for constructing the ModuleDepGraph incrementally.
class ModuleDepGraphBuilder {
public:
  ModuleDepGraphBuilder(StringRef ClangProgramPath)
      : ClangProgramPath(ClangProgramPath) {}

  /// Adds the scan result of a single scan to the ModuleDepGraph.
  void addScanResult(tooling::dependencies::TranslationUnitDeps TUDeps,
                     StringRef FileName);

  /// Takes and returns the constructured ModuleDepGraph.
  ModuleDepGraph takeGraph();

private:
  ModuleDepGraph MDG;

  StringRef ClangProgramPath;

  // Maps from module name to the corresponding node to avoid repeated graph
  // walks to find a module.
  llvm::StringMap<ClangModuleNode *> ClangModuleMap;
  llvm::StringMap<CXXNamedModuleNode *> StandardCXXModuleMap;

  /// Creates a new NonModuleTUNode.
  NonModuleTUNode *
  createNonModuleTUNode(std::string FilePath,
                        std::vector<tooling::dependencies::Command> Commands);

  /// Finds an existing Clang module node or creates a new one.
  /// If no existing module is found, an incomplete node witoutCommandLine is
  /// created.
  ClangModuleNode *getOrCreateClangModule(std::string ModuleName);

  /// Finds an existing Clang named module node or creates a new one.
  /// If an an incomplete node with only the module name is found, this resolves
  /// the CommandLine.
  ClangModuleNode *
  getOrCreateClangModule(std::string ModuleName,
                         std::vector<tooling::dependencies::Command> Commands);

  /// Finds an existing C++ named module node or creates a new one.
  /// If no existing module is found, an incomplete node without InputFile or
  /// CommandLine is created.
  CXXNamedModuleNode *getOrCreateCXXModule(std::string ModuleName);

  /// Finds an existing standard C++ module node or creates a new one.
  /// If an an incomplete node with only the module name is found, this resolves
  /// the InputFile and CommandLine.
  CXXNamedModuleNode *getOrCreateStandardCXXModule(
      std::string ModuleName, StringRef InputFile,
      std::vector<tooling::dependencies::Command> Commands);
};

NonModuleTUNode *ModuleDepGraphBuilder::createNonModuleTUNode(
    std::string FilePath,
    std::vector<tooling::dependencies::Command> Commands) {
  auto *NewNode = MDG.MakeWithBumpAlloc<NonModuleTUNode>(std::move(FilePath));
  MDG.addNode(*NewNode);
  return NewNode;
}

ClangModuleNode *
ModuleDepGraphBuilder::getOrCreateClangModule(std::string ModuleName) {
  auto It = ClangModuleMap.find(ModuleName);
  if (It != ClangModuleMap.end())
    return It->getValue();

  auto *NewNode = MDG.MakeWithBumpAlloc<ClangModuleNode>(std::move(ModuleName));
  MDG.addNode(*NewNode);
  ClangModuleMap.try_emplace(NewNode->getModuleName(), NewNode);
  return NewNode;
}

ClangModuleNode *ModuleDepGraphBuilder::getOrCreateClangModule(
    std::string ModuleName,
    std::vector<tooling::dependencies::Command> Commands) {
  auto *Node = getOrCreateClangModule(std::move(ModuleName));
  if (Node->getCommands().empty())
    Node->setCommands(std::move(Commands));
  return Node;
}

CXXNamedModuleNode *
ModuleDepGraphBuilder::getOrCreateCXXModule(std::string ModuleName) {
  auto It = StandardCXXModuleMap.find(ModuleName);
  if (It != StandardCXXModuleMap.end())
    return It->getValue();

  auto *NewNode =
      MDG.MakeWithBumpAlloc<CXXNamedModuleNode>(std::move(ModuleName), "");
  MDG.addNode(*NewNode);
  StandardCXXModuleMap.try_emplace(NewNode->getModuleName(), NewNode);
  return NewNode;
}

CXXNamedModuleNode *ModuleDepGraphBuilder::getOrCreateStandardCXXModule(
    std::string ModuleName, StringRef InputFile,
    std::vector<tooling::dependencies::Command> Commands) {
  auto *Node = getOrCreateCXXModule(std::move(ModuleName));
  Node->setInputFile(InputFile);
  Node->setCommands(std::move(Commands));
  return Node;
}

void ModuleDepGraphBuilder::addScanResult(
    tooling::dependencies::TranslationUnitDeps TUDeps, StringRef FileName) {
  for (auto &MD : TUDeps.ModuleGraph) {
    std::vector<std::string> BuildArgs = MD.getBuildArguments();
    std::vector<tooling::dependencies::Command> Cmds = {
        {ClangProgramPath.str(), std::move(BuildArgs)}};
    auto *Node =
        getOrCreateClangModule(std::move(MD.ID.ModuleName), std::move(Cmds));
    for (const auto &Dep : MD.ClangModuleDeps) {
      auto *DepNode = getOrCreateClangModule(std::move(Dep.ModuleName));
      if (Node->hasEdgeTo(*DepNode))
        continue;
      auto *Edge = MDG.MakeWithBumpAlloc<MDGEdge>(*DepNode);
      MDG.connect(*Node, *DepNode, *Edge);
    }
  }

  MDGNode *CurNode = nullptr;
  if (!TUDeps.ID.ModuleName.empty()) {
    CurNode = getOrCreateStandardCXXModule(
        std::move(TUDeps.ID.ModuleName), FileName, std::move(TUDeps.Commands));
  } else {
    CurNode = createNonModuleTUNode(FileName.str(), std::move(TUDeps.Commands));
  }

  for (const auto &CXXNamedMDName : TUDeps.NamedModuleDeps) {
    auto *DepNode = getOrCreateCXXModule(CXXNamedMDName);
    auto *Edge = MDG.MakeWithBumpAlloc<MDGEdge>(*DepNode);
    MDG.connect(*CurNode, *DepNode, *Edge);
  }

  for (auto &MD : TUDeps.ClangModuleDeps) {
    auto *DepNode = getOrCreateClangModule(MD.ModuleName);
    auto *Edge = MDG.MakeWithBumpAlloc<MDGEdge>(*DepNode);
    MDG.connect(*CurNode, *DepNode, *Edge);
  }
}

ModuleDepGraph ModuleDepGraphBuilder::takeGraph() { return std::move(MDG); }

ModuleDepGraph
buildModuleDepGraph(SmallVectorImpl<TranslationUnitDeps> &&TUDepsList,
                    Driver::InputList Inputs, StringRef ClangProgramPath) {
  ModuleDepGraphBuilder Builder(ClangProgramPath);
  for (auto &&[TUDeps, Input] : llvm::zip(TUDepsList, Inputs))
    Builder.addScanResult(TUDeps, Input.second->getSpelling());
  return Builder.takeGraph();
}

} // namespace dependencies
} // namespace driver
} // namespace clang

//===----------------------------------------------------------------------===//
// Module Dependency Graph: GraphTraits specialization
//===----------------------------------------------------------------------===//

using namespace clang::driver::dependencies;

namespace llvm {
void GraphWriter<const ModuleDepGraph *>::writeNodes() {
  auto IsNodeVisible = [&](NodeRef N) { return !DTraits.isNodeHidden(N, G); };
  auto VisibleNodeRange = llvm::make_filter_range(nodes(G), IsNodeVisible);
  SmallVector<NodeRef, 0> VisibleNodes(VisibleNodeRange);
  writeNodeDeclarations(VisibleNodes);
  writeNodeRelations(VisibleNodes);
}

void GraphWriter<const ModuleDepGraph *>::writeNodeDeclarations(
    ArrayRef<NodeRef> Nodes) {
  for (const auto &Node : Nodes)
    writeNodeDeclaration(Node);
  O << "\n";
}

void GraphWriter<const ModuleDepGraph *>::writeNodeDeclaration(NodeRef Node) {
  std::string NodeLabel;
  switch (Node->getKind()) {
  case MDGNode::NodeKind::ClangModule: {
    const auto *ClangModule = static_cast<const ClangModuleNode *>(Node);
    NodeLabel = "{ Type: Clang Module | Provides: \\\"" +
                ClangModule->getModuleName().str() + "\\\" }";
    break;
  }
  case MDGNode::NodeKind::CXXNamedModule: {
    const auto *CXXModuleNode = static_cast<const CXXNamedModuleNode *>(Node);
    NodeLabel = "{ Type: C++ Named Module";
    if (!CXXModuleNode->getInputFile().empty()) {
      NodeLabel +=
          " | Filename: \\\"" + CXXModuleNode->getInputFile().str() + "\\\"";
    } else
      NodeLabel += " | Filename: \\<unresolved\\>";
    NodeLabel +=
        " | Provides: \\\"" + CXXModuleNode->getModuleName().str() + "\\\" }";
    break;
  }
  case MDGNode::NodeKind::NonModuleTU: {
    const auto *TUNode = static_cast<const NonModuleTUNode *>(Node);
    NodeLabel = "{ Type: Default TU | Filename: \\\"" +
                TUNode->getInputFile().str() + "\\\" }";
    break;
  }
  }

  const auto NodeIdentifierLabel = DTraits.getNodeIdentifierLabel(Node, G);
  NodeIDLabels[Node] = NodeIdentifierLabel;
  O << "\t\"" << DOT::EscapeString(NodeIdentifierLabel) << "\" [ label = \""
    << NodeLabel << "\" ];\n";
}

void GraphWriter<const ModuleDepGraph *>::writeNodeRelations(
    ArrayRef<NodeRef> Nodes) {
  for (const auto &Node : Nodes)
    writeNodeRelation(Node);
}

void GraphWriter<const ModuleDepGraph *>::writeNodeRelation(NodeRef Node) {
  const auto SrcLabel = NodeIDLabels.lookup(Node);
  for (const auto *NodeIT = GTraits::child_edge_begin(Node);
       NodeIT != GTraits::child_edge_end(Node); ++NodeIT) {
    const auto *Edge = *NodeIT;
    const auto *TargetNode = GTraits::MDGGetTargetNode(Edge);
    const auto DestLabel = NodeIDLabels.lookup(TargetNode);
    if (SrcLabel.empty() || DestLabel.empty())
      continue;
    O << "\t\"" << DOT::EscapeString(SrcLabel) << "\" -> \""
      << DOT::EscapeString(DestLabel) << "\";\n";
  }
}

} // namespace llvm
