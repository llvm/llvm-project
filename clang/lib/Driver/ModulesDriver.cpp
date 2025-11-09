//===- DependencyScanner.cpp - Module dependency discovery ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Driver/ModulesDriver.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticDriver.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/InputInfo.h"
#include "clang/Driver/Job.h"
#include "clang/Driver/Tool.h"
#include "clang/Driver/Types.h"
#include "clang/Lex/Lexer.h"
#include "clang/Tooling/DependencyScanning/DependencyScanningTool.h"
#include "clang/Tooling/DependencyScanning/DependencyScanningWorker.h"
#include "clang/Tooling/DependencyScanning/ModuleDepCollector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/DirectedGraph.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/TargetParser/Host.h"
#include <atomic>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using namespace clang;
using namespace clang::driver;
using namespace llvm;
using namespace llvm::opt;

namespace clang::tooling {
namespace deps = dependencies;
} // namespace clang::tooling

using OwnedJobList = SmallVector<std::unique_ptr<Command>, 4>;

//===----------------------------------------------------------------------===//
// Check: Enable -fmodules-driver implicitly
//===----------------------------------------------------------------------===//

namespace clang::driver::modules {

/// Returns true if any input is a `.cppm` file.
static bool hasCXXModuleInputType(const InputList &Inputs) {
  const auto IsTypeCXXModule = [](const auto &Input) -> bool {
    const auto TypeID = Input.first;
    return (TypeID == types::TY_CXXModule);
  };
  return llvm::any_of(Inputs, IsTypeCXXModule);
}

/// Scans the leading lines of the C++ source inputs to detect C++20 module
/// usage.
///
/// \returns true if module usage is detected, false otherwise, or an error on
/// failure to read the input source.
static llvm::ErrorOr<bool>
ScanInputsForCXX20ModulesUsage(const InputList &Inputs,
                               llvm::vfs::FileSystem &VFS,
                               DiagnosticsEngine &Diags) {
  const auto CXXInputs = llvm::make_filter_range(
      Inputs, [](const auto &Input) { return types::isCXX(Input.first); });
  for (const auto &Input : CXXInputs) {
    StringRef Filename = Input.second->getSpelling();
    auto ErrOrBuffer = VFS.getBufferForFile(Filename);
    if (!ErrOrBuffer)
      return ErrOrBuffer.getError();
    const auto Buffer = std::move(*ErrOrBuffer);

    if (scanInputForCXX20ModulesUsage(Buffer->getBuffer())) {
      Diags.Report(diag::remark_found_cxx20_module_usage) << Filename;
      return true;
    }
  }
  return false;
}

/// Checks if the -fmodules-driver feature should be implicitly enabled for this
/// compilation.
///
/// The -fmodules-driver feature should be implicitly enabled iff (1) any input
/// makes used of C++20 named modules; and (2) there are more than two source
/// input files.
///
/// \returns true if the -fmodules-driver feature should be enabled, false
/// otherwise.
bool shouldEnableModulesDriver(const InputList &Inputs,
                               llvm::vfs::FileSystem &VFS,
                               DiagnosticsEngine &Diags) {
  if (Inputs.size() < 2)
    return false;

  bool UsesCXXModules = hasCXXModuleInputType(Inputs);
  if (UsesCXXModules)
    return true;

  const auto ErrOrScanResult =
      ScanInputsForCXX20ModulesUsage(Inputs, VFS, Diags);
  if (!ErrOrScanResult) {
    Diags.Report(diag::err_cannot_open_file)
        << ErrOrScanResult.getError().message();
  }
  return *ErrOrScanResult;
}

/// Builds the a C++ named module input for \c InputFile and adds it to \c Args.
static void addCXXModuleInput(InputList &Inputs, DerivedArgList &Args,
                              const OptTable &Opts, StringRef InputFile) {
  Arg *A = new Arg(Opts.getOption(options::OPT_INPUT), InputFile,
                   Args.getBaseArgs().MakeIndex(InputFile),
                   Args.getBaseArgs().MakeArgString(InputFile));
  Args.AddSynthesizedArg(A);
  A->claim();
  Inputs.push_back(std::make_pair(types::TY_CXXModule, A));
}

/// Parses the std modules manifest and builds the inputs for the discovered
/// std modules.
///
/// \returns true if the modules were added, false failure to read/parse the
/// manifest (with diagnostics reported using the drivers DiagnosticEngine).
bool ensureNamedModuleStdLibraryInputs(Compilation &C, InputList &Inputs) {
  const auto &Driver = C.getDriver();
  auto &Diags = Driver.getDiags();

  const auto ManifestPath =
      Driver.GetStdModuleManifestPath(C, C.getDefaultToolChain());
  Diags.Report(diag::remark_std_module_manifest_path) << ManifestPath;
  if (ManifestPath == "<NOT PRESENT>")
    return false;

  llvm::SmallString<256> ManifestDir(ManifestPath);
  llvm::sys::path::remove_filename(ManifestDir);

  auto MemBufOrErr = llvm::MemoryBuffer::getFile(ManifestPath);
  if (!MemBufOrErr) {
    Diags.Report(diag::err_cannot_open_file)
        << MemBufOrErr.getError().message();
    return false;
  }
  const auto MemBuf = std::move(*MemBufOrErr);

  auto ParsedJsonOrErr = llvm::json::parse(MemBuf->getBuffer());
  if (!ParsedJsonOrErr) {
    Diags.Report(diag::err_failed_parse_modules_manifest_json);
    llvm::consumeError(ParsedJsonOrErr.takeError());
    return false;
  }
  const auto ParsedJson = std::move(*ParsedJsonOrErr);

  const auto *ModulesInfoList = ParsedJson.getAsObject()->getArray("modules");
  if (!ModulesInfoList)
    return false;

  const auto Opts = Driver.getOpts();
  auto &Args = C.getArgs();
  for (const auto &Entry : *ModulesInfoList) {
    const auto *ModuleInfoObj = Entry.getAsObject();
    if (!ModuleInfoObj)
      return false;

    auto IsStdLib = ModuleInfoObj->getBoolean("is-std-library");
    if (!IsStdLib && !*IsStdLib)
      continue;

    if (auto SourcePath = ModuleInfoObj->getString("source-path")) {
      SmallString<248> AbsSourcePath(ManifestDir);
      llvm::sys::path::append(AbsSourcePath, *SourcePath);
      addCXXModuleInput(Inputs, Args, Opts, AbsSourcePath);
    }
  }

  return true;
}

} // namespace clang::driver::modules

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
  if (!sys::path::is_absolute(Abs)) {
    if (const auto &CWD =
            SrcMgr.getFileManager().getFileSystemOpts().WorkingDir;
        !CWD.empty())
      sys::fs::make_absolute(CWD, Abs);
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

  SmallVector<CharSourceRange, 0> TranslatedRanges;
  TranslatedRanges.reserve(StandaloneDiag.Ranges.size());
  transform(StandaloneDiag.Ranges, std::back_inserter(TranslatedRanges),
            ConvertOffsetRange);

  SmallVector<FixItHint, 0> TranslatedFixIts;
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
/// point in building the compilation, in which case the StandaloneDiagReporter
/// supplies its own.
class StandaloneDiagReporter {
public:
  explicit StandaloneDiagReporter(DiagnosticsEngine &Diags) : Diags(Diags) {
    if (!Diags.hasSourceManager()) {
      FileSystemOptions Opts;
      Opts.WorkingDir = ".";
      OwnedFileMgr = makeIntrusiveRefCnt<FileManager>(std::move(Opts));
      OwnedSrcMgr = makeIntrusiveRefCnt<SourceManager>(Diags, *OwnedFileMgr);
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
    DiagnosticConsumer::HandleDiagnostic(Level, Info);
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
// Dependency Scan
//===----------------------------------------------------------------------===//

namespace {
/// A simple dependency action controller that only provides module lookup for
/// Clang modules.
class ModuleLookupActionController
    : public tooling::deps::DependencyActionController {
public:
  ModuleLookupActionController(StringRef TempDir) : TempDir(TempDir) {}

  std::string
  lookupModuleOutput(const tooling::deps::ModuleDeps &MD,
                     tooling::deps::ModuleOutputKind Kind) override {
    if (Kind == tooling::deps::ModuleOutputKind::ModuleFile)
      return constructPCMPath(MD.ID);
    // Driver command lines which cause this should be handled either in
    // Driver::handleArguments and rejected or in
    // buildCommandLineForDummyDriver and modified.
    llvm::reportFatalInternalError(
        "call to lookupModuleOutput with unexpected ModuleOutputKind");
  }

private:
  SmallString<128> TempDir;

  std::string constructPCMPath(const tooling::deps::ModuleID &ID) const {
    SmallString<256> ExplicitPCMPath(TempDir);
    llvm::sys::path::append(ExplicitPCMPath, Twine(ID.ModuleName) + "-" +
                                                 ID.ContextHash + ".pcm");
    return std::string(ExplicitPCMPath);
  }
};
} // namespace

/// Returns the full -cc1 command line (incl. executable) for a driver command.
static std::vector<std::string> buildCC1CommandLine(const Command &Cmd) {
  const auto &CmdArgList = Cmd.getArguments();
  std::vector<std::string> Out;
  Out.reserve(CmdArgList.size() + 1);
  Out.emplace_back(Cmd.getExecutable());
  for (const auto &Arg : CmdArgList)
    Out.emplace_back(Arg);
  return Out;
}

/// Scans the full dependencies and Clang module graphs for all user inputs in
/// \c ScanInputCmds.
///
/// The scan input commands are expected to be ordered as:
///   [ user input jobs..., std module job, std.compat module job ]
/// The user input jobs are always scanned and the \c std and \c std.compat
/// jobs are scanned on demand if any user input imports them.
///
/// \returns a list of TranslationUnitDeps on succes or an empty list on
/// failure.
static SmallVector<tooling::deps::TranslationUnitDeps, 0>
scanDependencies(const ArrayRef<std::unique_ptr<Command>> ScanInputCmds,
                 DiagnosticsEngine &Diags, StringRef TempDir) {
  // The last 2 trailing jobs contain the std modules, always in order of first
  // std and then std.compat.
  // If the user does not import a standard library module, no scan is performed
  // for it, and the corresponding job is later removed from the job list.
  const auto InputCount = ScanInputCmds.size();
  const auto UserInputCount = InputCount - 2;

  llvm::DefaultThreadPool Pool;
  llvm::ThreadPoolTaskGroup UserInputTaskGroup(Pool);
  llvm::ThreadPoolTaskGroup StdModulesTaskGroup(Pool);

  enum class SeenStdModulesState : uint8_t {
    None = 0,
    Std = 1,
    StdAndStdCompat = 2
  };
  std::atomic<SeenStdModulesState> SeenStdModules = SeenStdModulesState::None;

  // Helper to update the required std module imports after scanning a user
  // input.
  auto UpdateStdImportFlags =
      [&](const tooling::deps::TranslationUnitDeps &Deps) {
        if (SeenStdModules.load(std::memory_order_relaxed) ==
            SeenStdModulesState::StdAndStdCompat)
          return;

        for (const auto &NamedDep : Deps.NamedModuleDeps) {
          if (NamedDep == "std.compat") {
            SeenStdModules.store(SeenStdModulesState::StdAndStdCompat,
                                 std::memory_order_relaxed);
            return;
          }
          if (NamedDep == "std") {
            // If this fails, the state is already set to Std or
            // StdAndStdCompat.
            auto Expected = SeenStdModulesState::None;
            (void)SeenStdModules.compare_exchange_strong(
                Expected, SeenStdModulesState::Std, std::memory_order_relaxed,
                std::memory_order_relaxed);
          }
        }
      };

  std::atomic<size_t> NextUserInputIdx = 0;
  std::atomic<size_t> NextStdInputIdx = UserInputCount;

  // Helper to get the next index to process for a worker.
  // Workers in the StdModulesTaskGroup wait until all user inputs have finished
  // scanning, and then scan only the required standard library modules.
  auto GetNextInputIndex =
      [&](bool IsStdModulesWorker) -> std::optional<size_t> {
    const size_t CurUserInputIdx =
        NextUserInputIdx.fetch_add(1, std::memory_order_relaxed);
    if (CurUserInputIdx < UserInputCount)
      return CurUserInputIdx;

    if (IsStdModulesWorker) {
      // Wait for all scans on user inputs to finish before reading the import
      // state.
      UserInputTaskGroup.wait();

      // Because std.compat is always the last list item, and import std.compat
      // implies import std,  we can simply extend iteration based the required
      // imports.
      const size_t ExtraCount =
          static_cast<size_t>(SeenStdModules.load(std::memory_order_relaxed));
      const size_t CurStdInputIdx =
          NextStdInputIdx.fetch_add(1, std::memory_order_relaxed);
      if (CurStdInputIdx < UserInputCount + ExtraCount)
        return CurStdInputIdx;
    }

    return std::nullopt;
  };

  const IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS =
      llvm::vfs::createPhysicalFileSystem();
  tooling::deps::DependencyScanningService Service(
      tooling::deps::ScanningMode::DependencyDirectivesScan,
      tooling::deps::ScanningOutputFormat::Full);

  SmallVector<tooling::deps::TranslationUnitDeps, 0> TUDepsList(InputCount);
  SmallVector<SmallVector<StandaloneDiagnostic, 0>, 0> DiagLists(InputCount);
  std::atomic<bool> HasError = false;

  auto RunScanningWorker = [&](bool IsStdModulesTask) -> void {
    tooling::deps::DependencyScanningWorker Worker(Service, FS);
    DenseSet<tooling::deps::ModuleID> AlreadySeen;

    while (const auto MaybeIndex = GetNextInputIndex(IsStdModulesTask)) {
      const auto Index = *MaybeIndex;
      const auto &Cmd = *ScanInputCmds[Index];
      const auto CommandLine = buildCC1CommandLine(Cmd);

      ModuleLookupActionController LookupController(TempDir);
      StandaloneDiagCollector DiagConsumer;
      tooling::deps::FullDependencyConsumer DepsConsumer(AlreadySeen);

      const bool Success =
          Worker.computeDependencies(/*CWD*/ ".", CommandLine, DepsConsumer,
                                     LookupController, DiagConsumer);
      if (!Success)
        HasError = true;

      DiagLists[Index] = DiagConsumer.takeDiags();
      TUDepsList[Index] = DepsConsumer.takeTranslationUnitDeps();

      // Check if we additionally need to scan std.cppm or std.compat.cppm.
      if (Index < UserInputCount)
        UpdateStdImportFlags(TUDepsList[Index]);
    }
  };

  // TODO: This will always make the StdModulesTaskGroup have one worker, but
  // scan up to two files. For 4 or more inputs, it would make sense to have two
  // StdModulesTaskGroup workers.
  if (UserInputCount > 3) {
    const size_t Concurrency =
        std::min(static_cast<size_t>(Pool.getMaxConcurrency()), UserInputCount);
    for (size_t I = 0; I < Concurrency - 1; ++I)
      Pool.async(UserInputTaskGroup, [&]() { RunScanningWorker(false); });
  }
  Pool.async(StdModulesTaskGroup, [&]() { RunScanningWorker(true); });
  Pool.wait();

  // Report diagnostics in the original source input order.
  const StandaloneDiagReporter DiagReporter(Diags);
  for (auto &StandaloneDiagList : DiagLists)
    for (auto &StandaloneDiag : StandaloneDiagList)
      DiagReporter.Report(std::move(StandaloneDiag));

  if (HasError)
    return {};

  // Trim output to omit entries for std library modules which weren't
  // imported.
  const size_t UnusedStdModuleImports =
      2 - static_cast<size_t>(SeenStdModules.load(std::memory_order_relaxed));
  TUDepsList.truncate(TUDepsList.size() - UnusedStdModuleImports);
  return TUDepsList;
}

//===----------------------------------------------------------------------===//
// Module Dependency Graph
//===----------------------------------------------------------------------===//

namespace {

class MDGNode;
class MDGEdge;
using MDGNodeBase = DGNode<MDGNode, MDGEdge>;
using MDGEdgeBase = DGEdge<MDGNode, MDGEdge>;
using ModuleDepGraphBase = DirectedGraph<MDGNode, MDGEdge>;

/// Abstract base class for all node kinds in the module dependency graph.
class MDGNode : public MDGNodeBase {
public:
  enum class NodeKind {
    Root,
    ClangModule,
    NamedCXXModule,
    NonModule,
  };

  explicit MDGNode(NodeKind Kind) : Kind(Kind) {}
  virtual ~MDGNode() = 0;

  /// Returns this node's kind.
  NodeKind getKind() const { return Kind; }

private:
  NodeKind Kind;
};

MDGNode::~MDGNode() {}

/// Represents the root node of the module dependency graph.
///
/// The root node only serves as an entry point for graph traversal.
/// It should have an edge to each node that would otherwise have no incoming
/// edges, ensuring there is always a path from the root to any node in the
/// graph.
/// There should be exactly one such root node in a given graph.
class RootMDGNode final : public MDGNode {
public:
  RootMDGNode() : MDGNode(NodeKind::Root) {}

  /// Define classof to be able to use isa<>, cast<>, dyn_cast<>, etc.
  static bool classof(const MDGNode *N) {
    return N->getKind() == NodeKind::Root;
  }
};

/// Base class defining common functionality for nodes that represent a
/// translation unit from a command line source input.
class TranslationUnitBackedMDGNode : public MDGNode {
protected:
  explicit TranslationUnitBackedMDGNode(
      const NodeKind Kind, tooling::dependencies::TranslationUnitDeps &TUDeps,
      const size_t JobIndex)
      : MDGNode(Kind), JobIndex(JobIndex),
        NamedModuleDeps(std::move(TUDeps.NamedModuleDeps)),
        ClangModuleDeps(std::move(TUDeps.ClangModuleDeps)) {
    assert(!TUDeps.FileDeps.empty() &&
           "TUDeps has to have a corresponding source file");
    FileName = TUDeps.FileDeps.front();
    assert(TUDeps.Commands.size() == 1 &&
           "Generated multiple commands for a single -cc1 job?");
    BuildArgs = std::move(TUDeps.Commands.front().Arguments);
  }

public:
  /// Define classof to be able to use isa<>, cast<>, dyn_cast<>, etc.
  static bool classof(const MDGNode *N) {
    auto K = N->getKind();
    return K == NodeKind::NonModule || K == NodeKind::NamedCXXModule;
  }

  size_t JobIndex;
  std::string FileName;
  std::vector<std::string> NamedModuleDeps;
  std::vector<tooling::deps::ModuleID> ClangModuleDeps;
  std::vector<std::string> BuildArgs;
};

/// Subclass of MDGNode representing a translation unit that provides a C++20
/// named module.
class CXXNamedModuleMDGNode final : public TranslationUnitBackedMDGNode {
public:
  CXXNamedModuleMDGNode(tooling::dependencies::TranslationUnitDeps &TUDeps,
                        const size_t JobIndex)
      : TranslationUnitBackedMDGNode(NodeKind::NamedCXXModule, TUDeps,
                                     JobIndex),
        ModuleName(std::move(TUDeps.ID.ModuleName)) {}

  /// Define classof to be able to use isa<>, cast<>, dyn_cast<>, etc.
  static bool classof(const MDGNode *N) {
    return N->getKind() == MDGNode::NodeKind::NamedCXXModule;
  }

  std::string ModuleName;
};

/// Subclass of MDGNode representing a translation unit that does not provide
/// any module.
class NonModuleMDGNode final : public TranslationUnitBackedMDGNode {
public:
  NonModuleMDGNode(tooling::dependencies::TranslationUnitDeps &TUDeps,
                   const size_t JobIndex)
      : TranslationUnitBackedMDGNode(NodeKind::NonModule, TUDeps, JobIndex) {}

  /// Define classof to be able to use isa<>, cast<>, dyn_cast<>, etc.
  static bool classof(const MDGNode *N) {
    return N->getKind() == MDGNode::NodeKind::NonModule;
  }
};

/// Subclass of MDGNode representing a Clang module unit.
class ClangModuleMDGNode final : public MDGNode {
public:
  ClangModuleMDGNode(tooling::dependencies::ModuleDeps &MD,
                     const size_t ParentJobIndex)
      : MDGNode(NodeKind::ClangModule), ParentJobIndex(ParentJobIndex),
        ID(std::move(MD.ID)), ClangModuleDeps(std::move(MD.ClangModuleDeps)),
        BuildArgs(MD.getBuildArguments()) {}

  /// Define classof to be able to use isa<>, cast<>, dyn_cast<>, etc.
  static bool classof(const MDGNode *N) {
    return N->getKind() == MDGNode::NodeKind::ClangModule;
  }

  size_t ParentJobIndex;
  tooling::deps::ModuleID ID;
  std::vector<tooling::deps::ModuleID> ClangModuleDeps;
  std::vector<std::string> BuildArgs;
};

/// Represents an import relation in the module dependency graph, directed
/// from the imported module to the importer.
class MDGEdge : public MDGEdgeBase {
public:
  explicit MDGEdge(MDGNode &N) : MDGEdgeBase(N) {}
  MDGEdge() = delete;
};

class ModuleDepGraphBuilder;

/// A directed graph describing the discovered module dependency relations.
///
/// The graph owns all of its nodes and edges.
/// The graph's root node is initialized on construction.
class ModuleDepGraph : public ModuleDepGraphBase {
public:
  ModuleDepGraph() {
    Root = new (Alloc.Allocate(sizeof(RootMDGNode), alignof(RootMDGNode)))
        RootMDGNode();
  }

  MDGNode *getRoot() { return Root; }

  const MDGNode *getRoot() const { return Root; }

  // Gets the graph's allocator which should be used to allocate all of the
  // Graph's nodes and edges.
  BumpPtrAllocator &getAllocator() { return Alloc; }

private:
  friend class ModuleDepGraphBuilder;

  BumpPtrAllocator Alloc;
  RootMDGNode *Root = nullptr;
};

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Module Dependency Graph: GraphTraits specializations
//===----------------------------------------------------------------------===//

namespace llvm {
/// Non-const versions of the GraphTraits specializations for ModuleDepGraph.
template <> struct GraphTraits<MDGNode *> {
  using NodeRef = MDGNode *;

  static NodeRef MDGGetTargetNode(MDGEdgeBase *E) {
    return &E->getTargetNode();
  }

  using ChildIteratorType =
      mapped_iterator<MDGNode::iterator, decltype(&MDGGetTargetNode)>;
  using ChildEdgeIteratorType = MDGNode::iterator;

  static NodeRef getEntryNode(NodeRef N) { return N; }

  static ChildIteratorType child_begin(NodeRef N) {
    return ChildIteratorType(N->begin(), &MDGGetTargetNode);
  }

  static ChildIteratorType child_end(NodeRef N) {
    return ChildIteratorType(N->end(), &MDGGetTargetNode);
  }

  static ChildEdgeIteratorType child_edge_begin(NodeRef N) {
    return N->begin();
  }
  static ChildEdgeIteratorType child_edge_end(NodeRef N) { return N->end(); }
};

template <> struct GraphTraits<ModuleDepGraph *> : GraphTraits<MDGNode *> {
  using GraphRef = ModuleDepGraph *;
  using NodeRef = MDGNode *;

  using nodes_iterator = ModuleDepGraph::iterator;

  static NodeRef getEntryNode(GraphRef G) { return G->getRoot(); }

  static nodes_iterator nodes_begin(GraphRef G) { return G->begin(); }

  static nodes_iterator nodes_end(GraphRef G) { return G->end(); }
};

/// Const versions of the GraphTraits specializations for ModuleDepGraph.
template <> struct GraphTraits<const MDGNode *> {
  using NodeRef = const MDGNode *;

  static NodeRef MDGGetTargetNode(const MDGEdgeBase *E) {
    return &E->getTargetNode();
  }

  using ChildIteratorType =
      mapped_iterator<MDGNode::const_iterator, decltype(&MDGGetTargetNode)>;
  using ChildEdgeIteratorType = MDGNode::const_iterator;

  static NodeRef getEntryNode(NodeRef N) { return N; }

  static ChildIteratorType child_begin(NodeRef N) {
    return ChildIteratorType(N->begin(), &MDGGetTargetNode);
  }

  static ChildIteratorType child_end(NodeRef N) {
    return ChildIteratorType(N->end(), &MDGGetTargetNode);
  }

  static ChildEdgeIteratorType child_edge_begin(NodeRef N) {
    return N->begin();
  }

  static ChildEdgeIteratorType child_edge_end(NodeRef N) { return N->end(); }
};

template <>
struct GraphTraits<const ModuleDepGraph *> : GraphTraits<const MDGNode *> {
  using GraphRef = const ModuleDepGraph *;
  using NodeRef = const MDGNode *;

  using nodes_iterator = ModuleDepGraph::const_iterator;

  static NodeRef getEntryNode(GraphRef G) { return G->getRoot(); }

  static nodes_iterator nodes_begin(GraphRef G) { return G->begin(); }

  static nodes_iterator nodes_end(GraphRef G) { return G->end(); }
};
} // namespace llvm

//===----------------------------------------------------------------------===//
// Module Dependency Graph: Graph Builder
//===----------------------------------------------------------------------===//

namespace {
/// Construction helper for the module dependency graph.
class ModuleDepGraphBuilder {
public:
  explicit ModuleDepGraphBuilder(DiagnosticsEngine &Diags) : Diags(Diags) {}

  bool build(SmallVectorImpl<tooling::deps::TranslationUnitDeps> &&ScanResults);

  ModuleDepGraph takeGraph() { return std::move(Graph); }

private:
  ModuleDepGraph Graph;
  DiagnosticsEngine &Diags;

  /// Lookup maps used for connecting the nodes.
  DenseMap<tooling::deps::ModuleID, ClangModuleMDGNode *> ClangModuleNodeMap;
  llvm::StringMap<CXXNamedModuleMDGNode *> CXXNamedModuleMap;

  /// Allocation helper using the graph's allocator.
  template <typename MDGComponent, typename... Args>
  MDGComponent *makeWithGraphAlloc(Args &&...args) {
    return new (
        Graph.Alloc.Allocate(sizeof(MDGComponent), alignof(MDGComponent)))
        MDGComponent(std::forward<Args>(args)...);
  }

  void addAllClangModuleNodes(
      SmallVectorImpl<tooling::deps::TranslationUnitDeps> &ScanResults);
  void addClangModuleGraph(tooling::deps::ModuleDepsGraph &ClangModuleGraph,
                           size_t ParentJob);
  void addClangModuleNode(tooling::deps::ModuleDeps &MD, size_t ParentJob);
  void connectClangModuleNodes();

  bool addAllTUBackedNodes(
      SmallVectorImpl<tooling::deps::TranslationUnitDeps> &ScanResults);
  bool addCXXNamedModuleNode(tooling::deps::TranslationUnitDeps &&TUDeps,
                             size_t Job);
  void addNonModuleNode(tooling::deps::TranslationUnitDeps &&TUDeps,
                        size_t Job);
  void connectTUBackedNodes();

  void addImportEdge(MDGNode &Imported, MDGNode &Importer);
};
} // anonymous namespace

bool ModuleDepGraphBuilder::build(
    SmallVectorImpl<tooling::deps::TranslationUnitDeps> &&ScanResults) {
  addAllClangModuleNodes(ScanResults);
  connectClangModuleNodes();
  if (!addAllTUBackedNodes(ScanResults))
    return false;
  connectTUBackedNodes();
  return true;
}

// Construct all Clang module nodes for this graph.
void ModuleDepGraphBuilder::addAllClangModuleNodes(
    SmallVectorImpl<tooling::deps::TranslationUnitDeps> &ScanResults) {
  for (size_t I = 0, E = ScanResults.size(); I < E; ++I) {
    addClangModuleGraph(ScanResults[I].ModuleGraph, I);
  }
}

/// Adds a new Clang module node every not yet seen Clang module in \c
/// MDGraph.
void ModuleDepGraphBuilder::addClangModuleGraph(
    tooling::deps::ModuleDepsGraph &ClangModuleGraph, size_t ParentJob) {
  for (auto &MD : ClangModuleGraph) {
    if (ClangModuleNodeMap.contains(MD.ID))
      continue;
    addClangModuleNode(MD, ParentJob);
  }
}

void ModuleDepGraphBuilder::addClangModuleNode(tooling::deps::ModuleDeps &MD,
                                               size_t ParentJob) {
  auto *Node = makeWithGraphAlloc<ClangModuleMDGNode>(MD, ParentJob);
  const auto [_, Inserted] = ClangModuleNodeMap.try_emplace(Node->ID, Node);
  assert(Inserted && "Duplicate nodes for a single Clang module!");
  Graph.addNode(*Node);
}

/// Interconnect all Clang module nodes.
void ModuleDepGraphBuilder::connectClangModuleNodes() {
  for (auto *Node : nodes(&Graph)) {
    auto *Importer = cast<ClangModuleMDGNode>(Node);

    if (Importer->ClangModuleDeps.empty()) {
      // No imports: connect to root for reachability and continue.
      addImportEdge(*Graph.getRoot(), *Importer);
      continue;
    }

    for (const auto &ID : Importer->ClangModuleDeps) {
      // The dependency scanning worker is expected to error if any required
      // dependency is not found.
      auto Found = ClangModuleNodeMap.find(ID);
      assert(Found != ClangModuleNodeMap.end() &&
             "Missing imported Clang module node");
      if (Found == ClangModuleNodeMap.end())
        continue;

      MDGNode *Imported = Found->second;
      addImportEdge(*Imported, *Importer);
    }
  }
}

// Construct all nodes for this graph, which represent the dependencies of a
// source input.
bool ModuleDepGraphBuilder::addAllTUBackedNodes(
    SmallVectorImpl<tooling::deps::TranslationUnitDeps> &ScanResults) {
  for (size_t I = 0, E = ScanResults.size(); I < E; ++I) {
    auto &TUDeps = ScanResults[I];
    if (TUDeps.ID.ModuleName.empty()) {
      // Non-module TU
      addNonModuleNode(std::move(TUDeps), I);
    } else {
      // C++ named module TU
      if (!addCXXNamedModuleNode(std::move(TUDeps), I))
        return false;
    }
  }
  return true;
}

/// Adds a node representing the non-module translation unit described by \c
/// TUDeps.
void ModuleDepGraphBuilder::addNonModuleNode(
    tooling::deps::TranslationUnitDeps &&TUDeps, size_t Job) {
  auto *Node = makeWithGraphAlloc<NonModuleMDGNode>(TUDeps, Job);
  Graph.addNode(*Node);
}

/// Adds a node representing the C++ named module unit described by \c
/// TUDeps.
///
/// \returns false on error if a node for the same module interface already
/// exists in the graph.
bool ModuleDepGraphBuilder::addCXXNamedModuleNode(
    tooling::deps::TranslationUnitDeps &&TUDeps, size_t Job) {
  CXXNamedModuleMDGNode *Node =
      makeWithGraphAlloc<CXXNamedModuleMDGNode>(TUDeps, Job);
  StringRef ModuleName = Node->ModuleName;
  const auto [It, Inserted] = CXXNamedModuleMap.try_emplace(ModuleName, Node);
  if (!Inserted) {
    StringRef ExistingFile = It->second->FileName;
    StringRef ThisFile = Node->FileName;
    Diags.Report(diag::err_mod_graph_named_module_redefinition)
        << ModuleName << ExistingFile << ThisFile;
    return false;
  }

  Graph.addNode(*Node);
  return true;
}

// Construct all nodes for this graph, which represent the dependencies of a
// source input.
void ModuleDepGraphBuilder::connectTUBackedNodes() {
  for (auto *Node : nodes(&Graph)) {
    if (auto *Importer = dyn_cast<TranslationUnitBackedMDGNode>(Node)) {
      if (Importer->NamedModuleDeps.empty() &&
          Importer->ClangModuleDeps.empty()) {
        addImportEdge(*Graph.getRoot(), *Importer);
        continue;
      }
      for (const auto &Name : Importer->NamedModuleDeps)
        if (auto *Imported = CXXNamedModuleMap.lookup(Name))
          addImportEdge(*Imported, *Importer);
      for (const auto &ID : Importer->ClangModuleDeps)
        if (auto *Imported = ClangModuleNodeMap.lookup(ID))
          addImportEdge(*Imported, *Importer);
    }
  }
}

/// Creates and adds an edge from \c Imported to \c Importer.
void ModuleDepGraphBuilder::addImportEdge(MDGNode &Imported,
                                          MDGNode &Importer) {
  auto *Edge = makeWithGraphAlloc<MDGEdge>(Importer);
  Imported.addEdge(*Edge);
}

/// Construct the module dependency graph for the given \c ScanResults.
///
/// \returns The constructed graph, or an std::nullopt on error, if
/// \c ScanResults contains conflicting module definitions.
static std::optional<ModuleDepGraph> buildModuleDepGraph(
    SmallVectorImpl<tooling::deps::TranslationUnitDeps> &&ScanResults,
    DiagnosticsEngine &Diags) {
  ModuleDepGraphBuilder Builder(Diags);
  if (!Builder.build(std::move(ScanResults)))
    return std::nullopt;
  return Builder.takeGraph();
}

//===----------------------------------------------------------------------===//
// Module Dependency Graph: GraphViz Output
//===----------------------------------------------------------------------===//

namespace llvm {
template <>
struct DOTGraphTraits<const ModuleDepGraph *> : DefaultDOTGraphTraits {
  explicit DOTGraphTraits(bool IsSimple = false)
      : DefaultDOTGraphTraits(IsSimple) {}

  static std::string getGraphName(const ModuleDepGraph *) {
    return "Module Dependency Graph";
  }

  static std::string getGraphProperties(const ModuleDepGraph *) {
    return "\tnode [shape=Mrecord];\n\tedge [dir=\"back\"];\n";
  }

  static bool isNodeHidden(const MDGNode *N, const ModuleDepGraph *) {
    return isa<RootMDGNode>(N);
  }

  static std::string getNodeIdentifier(const MDGNode *N,
                                       const ModuleDepGraph *) {
    SmallString<128> Buf;
    raw_svector_ostream OS(Buf);
    TypeSwitch<const MDGNode *>(N)
        .Case<ClangModuleMDGNode>([&](auto *N) { OS << N->ID.ModuleName; })
        .Case<CXXNamedModuleMDGNode>([&](auto *N) { OS << N->ModuleName; })
        .Case<NonModuleMDGNode>([&](auto *N) { OS << N->FileName; })
        .Default([](auto *) { llvm_unreachable("Unhandled MDGNode kind!"); });
    OS << " (" << getNodeKindStr(N->getKind()) << ")";

    return std::string(OS.str());
  }

  static std::string getNodeLabel(const MDGNode *N, const ModuleDepGraph *) {
    SmallString<128> Buf;
    raw_svector_ostream OS(Buf);
    OS << "Type: " << getNodeKindStr(N->getKind()) << " \\| ";

    auto PrintFilename = [](raw_ostream &OS, StringRef Filename) {
      OS << "Filename: " << Filename;
    };
    auto PrintModuleName = [](raw_ostream &OS, StringRef ModuleName) {
      OS << "Provides: " << ModuleName;
    };
    TypeSwitch<const MDGNode *>(N)
        .Case<const ClangModuleMDGNode>(
            [&](auto *N) { PrintModuleName(OS, N->ID.ModuleName); })
        .Case<const CXXNamedModuleMDGNode>([&](auto *N) {
          PrintModuleName(OS, N->ModuleName);
          OS << " \\| ";
          PrintFilename(OS, N->FileName);
        })
        .Case<const NonModuleMDGNode>(
            [&](auto *N) { PrintFilename(OS, N->FileName); })
        .Default([](auto *) {
          llvm::reportFatalInternalError("Unhandled MDGNode kind!");
        });

    return std::string(OS.str());
  }

private:
  static StringRef getNodeKindStr(MDGNode::NodeKind Kind) {
    using NodeKind = MDGNode::NodeKind;
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
class GraphWriter<const ModuleDepGraph *>
    : public GraphWriterBase<const ModuleDepGraph *,
                             GraphWriter<const ModuleDepGraph *>> {
public:
  using GraphType = const ModuleDepGraph *;
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

void GraphWriter<const ModuleDepGraph *>::writeNodes() {
  auto IsNodeVisible = [&](NodeRef N) { return !DTraits.isNodeHidden(N, G); };
  const auto VisibleNodeRange = make_filter_range(nodes(G), IsNodeVisible);
  const SmallVector<NodeRef, 0> VisibleNodes(VisibleNodeRange);

  writeNodeDefinitions(VisibleNodes);
  writeNodeRelations(VisibleNodes);
}

void GraphWriter<const ModuleDepGraph *>::writeNodeDefinitions(
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

void GraphWriter<const ModuleDepGraph *>::writeNodeRelations(
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

//===----------------------------------------------------------------------===//
// Modules Driver
//===----------------------------------------------------------------------===//

/// Returns true if a driver command is a viable dependency scan input.
/// We consider only clang -cc1 jobs whose inputs are all source inputs.
static bool isJobForDepScan(const Command &Cmd) {
  if (llvm::StringRef(Cmd.getCreator().getName()) != "clang")
    return false;
  auto IsSrcInput = [](const InputInfo &II) -> bool {
    return types::isSrcFile(II.getType());
  };
  return llvm::all_of(Cmd.getInputInfos(), IsSrcInput);
}

/// Partition \c Cmds into dependency scan input jobs and regular jobs.
static std::pair<OwnedJobList, OwnedJobList> partitionsCmds(JobList &&Cmds) {
  OwnedJobList ScanInputCmds, DependentCmds;
  for (auto &Job : Cmds.getJobs()) {
    if (isJobForDepScan(*Job))
      ScanInputCmds.push_back(std::move(Job));
    else
      DependentCmds.push_back(std::move(Job));
  }
  Cmds.getJobs().clear();
  return {std::move(ScanInputCmds), std::move(DependentCmds)};
}

/// Modifies \c ScanInputCmds and \c DependentCmds to delete the jobs for std
/// modules which were not imported.
static void pruneUnusedStdModuleJobs(
    SmallVectorImpl<std::unique_ptr<Command>> &ScanInputCmds,
    SmallVectorImpl<std::unique_ptr<Command>> &DependentCmds,
    SmallVector<tooling::deps::TranslationUnitDeps, 0> ScanOutputs) {
  const size_t StdModulesToRemove = ScanInputCmds.size() - ScanOutputs.size();
  if (StdModulesToRemove == 0)
    return;

  const size_t UnusedStdModulesFirst = ScanOutputs.size();
  const size_t UnusedStdModulesEnd = ScanInputCmds.size();

  // Remove the object files, created by Standard library module jobs which
  // are to be deleted, from the final linker job.
  llvm::StringSet<> LinkerInputsToRemove;
  for (size_t I = UnusedStdModulesFirst; I < UnusedStdModulesEnd; ++I) {
    for (StringRef Out : ScanInputCmds[I]->getOutputFilenames())
      LinkerInputsToRemove.insert(Out);
  }

  if (!LinkerInputsToRemove.empty() && !DependentCmds.empty()) {
    const auto &MaybeLinkJob = DependentCmds.back();
    if (MaybeLinkJob->getSource().getKind() ==
        Action::ActionClass::LinkJobClass) {
      const auto &LinkJobArguments = MaybeLinkJob->getArguments();
      ArgStringList NewArgList;
      NewArgList.reserve(LinkJobArguments.size());
      for (const char *Arg : LinkJobArguments) {
        if (LinkerInputsToRemove.contains(StringRef(Arg)))
          continue;
        NewArgList.push_back(Arg);
      }
      MaybeLinkJob->replaceArguments(NewArgList);
    }
  }

  // Delete the unused Standard library module jobs themselves.
  ScanInputCmds.erase(
      ScanInputCmds.begin() + static_cast<ptrdiff_t>(UnusedStdModulesFirst),
      ScanInputCmds.begin() + static_cast<ptrdiff_t>(UnusedStdModulesEnd));
}

/// Returns the list of topologically sorted nodes for \c Graph, excluding the
/// root node.
static SmallVector<MDGNode *, 0>
getTopologicallySortedNodes(ModuleDepGraph &Graph) {
  SmallVector<MDGNode *, 0> PostOrder;
  PostOrder.reserve(Graph.size());

  for (auto *N : llvm::post_order(&Graph))
    PostOrder.push_back(N);

  SmallVector<MDGNode *, 0> TopoSortedGraph = PostOrder;
  std::reverse(TopoSortedGraph.begin(), TopoSortedGraph.end());
  // Drop the root node!
  return SmallVector<MDGNode *, 0>(TopoSortedGraph.begin() + 1,
                                   TopoSortedGraph.end());
}

/// Replaces the build arguments for each job with those generated by the
/// dependency scan. For Clang modules, new jobs are created before.
static OwnedJobList installBuildArgsAndClangModuleJobs(
    Compilation &C, SmallVectorImpl<MDGNode *> &NodesInBuildOrder,
    OwnedJobList &ScanInputCmds) {
  OwnedJobList Out;
  Out.reserve(ScanInputCmds.size());

  auto InstallArgStrList = [&](Command &Cmd, ArrayRef<std::string> Args) {
    ArgStringList NewArgs;
    NewArgs.reserve(Args.size());

    auto &TCArgs = C.getArgsForToolChain(
        &Cmd.getCreator().getToolChain(), Cmd.getSource().getOffloadingArch(),
        Cmd.getSource().getOffloadingDeviceKind());
    for (const auto &S : Args)
      NewArgs.push_back(TCArgs.MakeArgString(S));

    Cmd.replaceArguments(NewArgs);
  };

  /// TODO: The 'linkBuildArguments' function can be implemented alot cleaner if
  /// we store the jobs inside of the graph instead of modifying their original
  /// index.
  for (size_t I = 0, E = NodesInBuildOrder.size(); I < E; ++I) {
    auto *Node = NodesInBuildOrder[I];
    TypeSwitch<MDGNode *>(Node)
        .Case<ClangModuleMDGNode>([&](auto *ClangModuleNode) {
          const auto &ParentJob =
              ScanInputCmds[ClangModuleNode->ParentJobIndex];
          auto ClangModuleJob = std::make_unique<Command>(*ParentJob);
          InstallArgStrList(*ClangModuleJob, ClangModuleNode->BuildArgs);
          ClangModuleNode->ParentJobIndex = I;
          Out.push_back(std::move(ClangModuleJob));
        })
        .Case<CXXNamedModuleMDGNode, NonModuleMDGNode>([&](auto *TUBackedNode) {
          auto Job = std::move(ScanInputCmds[TUBackedNode->JobIndex]);
          TUBackedNode->JobIndex = I;
          InstallArgStrList(*Job, TUBackedNode->BuildArgs);
          Out.push_back(std::move(Job));
        })
        .Default([](auto *) {
          llvm::reportFatalInternalError("Unhandled MDGNode kind!");
        });
  }
  return Out;
}

/// Returns the \c Argv as InputArgList. The returned InputArgList is only valid
/// for the lifetime of \c Argv.
/// TODO: The 'linkBuildArguments' function can be implemented alot cleaner if
/// we store the jobs inside of the graph instead of modifying their original
/// index.
static llvm::opt::InputArgList
parseInputArgList(llvm::opt::ArgStringList Argv) {
  const auto &Opts = getDriverOptTable();
  unsigned MissingArgIndex = 0, MissingArgCount = 0;
  return Opts.ParseArgs(Argv, MissingArgIndex, MissingArgCount,
                        llvm::opt::Visibility(options::CC1Option));
}

/// Modifies the job's command lines to properly output and pass C++ named
/// module dependencies to each other.
static void
linkBuildArguments(Compilation &C, ArrayRef<MDGNode *> NodesInBuildOrder,
                   SmallVectorImpl<std::unique_ptr<Command>> &OrderedCC1Jobs,
                   SmallVectorImpl<std::unique_ptr<Command>> &JobListTail) {
  // TODO: Use same temporary directory as used for the dependency scan.
  // TODO: Output files which are not associated with a job action do not get
  // cleaned up properly.
  auto constructPCMPath = [](Compilation &C, const StringRef &ModuleName) {
    return C.getDriver().CreateTempFile(C, ModuleName, "pcm");
  };
  // Propagate -fmodule-file=<module-name>=<module-path>
  for (auto *Node : NodesInBuildOrder) {
    if (auto *TUNode = llvm::dyn_cast<CXXNamedModuleMDGNode>(Node)) {
      StringRef ModuleOutput;
      auto &Job = OrderedCC1Jobs[TUNode->JobIndex];
      auto &Args = Job->getArguments();
      auto InputArgsList = parseInputArgList(Args);
      ModuleOutput = constructPCMPath(C, TUNode->ModuleName);
      auto &TCArgs =
          C.getArgsForToolChain(&Job->getCreator().getToolChain(),
                                Job->getSource().getOffloadingArch(),
                                Job->getSource().getOffloadingDeviceKind());
      // Always create an -fmodule-output to take control of the output path,
      // even for .cppm files, where this was already added.
      Args.push_back(
          TCArgs.MakeArgString(Twine("-fmodule-output=") + ModuleOutput));
      Args.push_back("-fmodules-reduced-bmi");
      // Hack: Hack to omit the warning for the std module, because I did not
      // get this working from the driver command line yet.
      Args.push_back("-Wno-reserved-module-identifier");

      // Propogate to dependent -cc1 commands in the graph
      for (auto *ChildNode : llvm::depth_first(Node)) {
        auto *ChildTUNode = llvm::cast<TranslationUnitBackedMDGNode>(ChildNode);
        if (ChildTUNode->JobIndex == TUNode->JobIndex)
          continue;
        auto &ChildJob = OrderedCC1Jobs[ChildTUNode->JobIndex];
        auto &TCArgs = C.getArgsForToolChain(
            &ChildJob->getCreator().getToolChain(),
            ChildJob->getSource().getOffloadingArch(),
            ChildJob->getSource().getOffloadingDeviceKind());
        ChildJob->getArguments().push_back(TCArgs.MakeArgString(
            Twine("-fmodule-file=") + TUNode->ModuleName + "=" + ModuleOutput));
      }

      // Propogate to dependent commands, which we not part of the dependency
      // scan.
      for (auto &TailCmd : JobListTail) {
        if (StringRef(TailCmd->getCreator().getName()) != "clang")
          continue;
        auto II = TailCmd->getInputInfos();
        if (II.empty())
          continue;
        if (II.front().isFilename() &&
            II.front().getFilename() == TUNode->FileName) {
          auto &TCArgs = C.getArgsForToolChain(
              &TailCmd->getCreator().getToolChain(),
              TailCmd->getSource().getOffloadingArch(),
              TailCmd->getSource().getOffloadingDeviceKind());
          TailCmd->getArguments().push_back(
              TCArgs.MakeArgString(Twine("-fmodule-file=") +
                                   TUNode->ModuleName + "=" + ModuleOutput));
        }
      }
    }
  }
}

namespace clang::driver::modules {

bool performDriverModuleBuild(Compilation &C, DiagnosticsEngine &Diags) {
  auto [ScanInputCmds, JobListTail] = partitionsCmds(std::move(C.getJobs()));

  // The directory containing all module artifacts.
  // TODO: Output files which are not associated with a job action do not get
  // cleaned up properly.
  const auto TempDir = C.getDriver().GetTemporaryDirectory("modules-driver");
  C.addTempFile(C.getArgs().MakeArgString(TempDir));

  auto ScanOutputs = scanDependencies(ScanInputCmds, Diags, TempDir);
  if (ScanOutputs.empty()) {
    Diags.Report(diag::err_failed_depdendency_scan);
    return false;
  }
  // If the Standard library modules are not needed, remove them fully.
  pruneUnusedStdModuleJobs(ScanInputCmds, JobListTail, ScanOutputs);

  auto DepGraph = buildModuleDepGraph(std::move(ScanOutputs), Diags);
  if (!DepGraph) {
    Diags.Report(diag::err_building_depdendency_graph);
    return false;
  }
  Diags.Report(diag::remark_printing_module_graph);
  if (!Diags.isLastDiagnosticIgnored())
    WriteGraph<const ModuleDepGraph *>(llvm::errs(), &(*DepGraph));

  auto NodesInBuildOrder = getTopologicallySortedNodes(*DepGraph);

  auto OrderedCC1Jobs =
      installBuildArgsAndClangModuleJobs(C, NodesInBuildOrder, ScanInputCmds);

  linkBuildArguments(C, NodesInBuildOrder, OrderedCC1Jobs, JobListTail);

  // Reconstruct the job list.
  C.getJobs().getJobs() = std::move(OrderedCC1Jobs);
  for (auto &NonScanJob : JobListTail)
    C.getJobs().getJobs().push_back(std::move(NonScanJob));
  return true;
}

} // namespace clang::driver::modules
