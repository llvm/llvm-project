//===--- Driver.cpp - Clang GCC Compatible Driver -------------------------===//
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
#include "clang/Basic/DiagnosticDriver.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Job.h"
#include "clang/Driver/Options.h"
#include "clang/Driver/Tool.h"
#include "clang/Lex/DependencyDirectivesScanner.h"
#include "clang/Lex/Lexer.h"
#include "clang/Tooling/DependencyScanning/DependencyScanningService.h"
#include "clang/Tooling/DependencyScanning/DependencyScanningTool.h"
#include "clang/Tooling/DependencyScanning/DependencyScanningWorker.h"
#include "clang/Tooling/DependencyScanning/ModuleDepCollector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/DirectedGraph.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/DOTGraphTraits.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/VirtualFileSystem.h"
#include <atomic>
#include <iterator>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <tuple>
#include <utility>

using namespace llvm::opt;

namespace clang::driver::modules {
using JobVector = JobList::list_type;

// The tooling::deps namespace has conflicting names with clang::driver, we
// therefore introduce only the required tooling::deps namespace members into
// this namespace.
using tooling::dependencies::DependencyActionController;
using tooling::dependencies::DependencyScanningService;
using tooling::dependencies::DependencyScanningWorker;
using tooling::dependencies::FullDependencyConsumer;
using tooling::dependencies::ModuleDeps;
using tooling::dependencies::ModuleDepsGraph;
using tooling::dependencies::ModuleID;
using tooling::dependencies::ModuleOutputKind;
using tooling::dependencies::ScanningMode;
using tooling::dependencies::ScanningOutputFormat;
using tooling::dependencies::TranslationUnitDeps;

/// Returns true if any source input signals C++ module usage.
static bool hasCXXNamedModuleInput(const InputList &Inputs) {
  const auto IsTypeCXXModule = [](const auto &Input) -> bool {
    const auto TypeID = Input.first;
    return (TypeID == types::TY_CXXModule || TypeID == types::TY_PP_CXXModule);
  };
  return any_of(Inputs, IsTypeCXXModule);
}

/// Scan the leading lines of each C++ source file until C++20 named module
/// usage is detected.
///
/// \returns true if module usage is detected, false otherwise, or a
/// llvm::FileError on read failure.
static Expected<bool> scanForCXXNamedModuleUsage(const InputList &Inputs,
                                                 llvm::vfs::FileSystem &VFS,
                                                 DiagnosticsEngine &Diags) {
  const auto CXXInputs = make_filter_range(
      Inputs, [](const InputTy &Input) { return types::isCXX(Input.first); });
  for (const auto &Input : CXXInputs) {
    auto Filename = Input.second->getSpelling();
    auto MemBufOrErr = VFS.getBufferForFile(Filename);
    if (!MemBufOrErr)
      return llvm::createFileError(Filename, MemBufOrErr.getError());
    const auto MemBuf = std::move(*MemBufOrErr);

    // Scan the buffer using the dependency directives scanner.
    if (clang::scanInputForCXXNamedModulesUsage(MemBuf->getBuffer())) {
      Diags.Report(diag::remark_found_cxx20_module_usage) << Filename;
      return true;
    }
  }
  return false;
}

Expected<bool> shouldUseModulesDriver(const InputList &Inputs,
                                      llvm::vfs::FileSystem &FS,
                                      DiagnosticsEngine &Diags) {
  if (Inputs.size() < 2)
    return false;
  if (hasCXXNamedModuleInput(Inputs))
    return true;
  return scanForCXXNamedModuleUsage(Inputs, FS, Diags);
}

static bool fromJSON(const llvm::json::Value &Params,
                     StdModuleManifest::LocalModuleArgs &LocalArgs,
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
  return O.map("modules", Manifest.ModuleEntries);
}

/// Parses the Standard library module manifest from \c Buffer.
///
/// The source file paths listed in the manifest are relative to its own
/// path.
static Expected<StdModuleManifest> parseStdModuleManifest(StringRef Buffer) {
  auto ParsedJsonOrErr = llvm::json::parse(Buffer);
  if (!ParsedJsonOrErr)
    return ParsedJsonOrErr.takeError();

  StdModuleManifest Manifest;
  llvm::json::Path::Root Root;
  if (!fromJSON(*ParsedJsonOrErr, Manifest, Root))
    return Root.getError();

  return Manifest;
}

/// Converts all file paths in \c Manifest from paths relative to
/// \c ManifestPath (the manifest's location itself) to absolute.
static void makeStdModuleManifestPathsAbsolute(StdModuleManifest &Manifest,
                                               StringRef ManifestPath) {
  SmallString<124> ManifestDir(ManifestPath);
  llvm::sys::path::remove_filename(ManifestDir);

  SmallString<256> TempPath;
  auto ensureAbsolutePath = [&](std::string &Path) {
    if (llvm::sys::path::is_absolute(Path))
      return;
    TempPath = ManifestDir;
    llvm::sys::path::append(TempPath, Path);
    llvm::sys::path::remove_dots(TempPath, true);
    Path = std::string(TempPath);
  };

  for (auto &ModuleEntry : Manifest.ModuleEntries) {
    ensureAbsolutePath(ModuleEntry.SourcePath);
    if (!ModuleEntry.LocalArgs)
      continue;
    for (auto &IncludeDir : ModuleEntry.LocalArgs->SystemIncludeDirs)
      ensureAbsolutePath(IncludeDir);
  }
}

Expected<StdModuleManifest> readStdModuleManifest(StringRef ManifestPath,
                                                  llvm::vfs::FileSystem &VFS) {
  auto MemBufOrErr = VFS.getBufferForFile(ManifestPath);
  if (!MemBufOrErr)
    return llvm::createFileError(ManifestPath, MemBufOrErr.getError());
  const auto MemBuf = std::move(*MemBufOrErr);

  auto ManifestOrErr = parseStdModuleManifest(MemBuf->getBuffer());
  if (!ManifestOrErr)
    return ManifestOrErr.takeError();
  auto Manifest = std::move(*ManifestOrErr);

  // All paths in the manifest are relative to \c ManifestPath.
  // Make them absolute.
  makeStdModuleManifestPathsAbsolute(Manifest, ManifestPath);

  return Manifest;
}

/// Appends a compilation input for the given \c Entry of the Standard library
/// module manifest.
static void
appendStdModuleManifestInput(const StdModuleManifest::Module &ModuleEntry,
                             Compilation &C, InputList &Inputs) {
  auto &Args = C.getArgs();
  const auto &Opts = C.getDriver().getOpts();

  C.getDriver().DiagnoseInputExistence(Args, ModuleEntry.SourcePath,
                                       types::TY_CXXModule,
                                       /*TypoCorrect=*/false);

  auto *A = new Arg(Opts.getOption(options::OPT_INPUT), ModuleEntry.SourcePath,
                    Args.getBaseArgs().MakeIndex(ModuleEntry.SourcePath),
                    Args.getBaseArgs().MakeArgString(ModuleEntry.SourcePath));
  Args.AddSynthesizedArg(A);
  A->claim();
  Inputs.emplace_back(types::TY_CXXModule, A);
}

void buildStdModuleManifestInputs(const StdModuleManifest &Manifest,
                                  Compilation &C, InputList &Inputs) {
  for (const auto &Module : Manifest.ModuleEntries)
    appendStdModuleManifestInput(Module, C, Inputs);
}

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

using StandaloneDiagList = SmallVector<StandaloneDiagnostic, 0>;
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
  if (const auto &CWD = SrcMgr.getFileManager().getFileSystemOpts().WorkingDir;
      !CWD.empty())
    llvm::sys::path::make_absolute(CWD, Abs);
  return std::string(Abs.str());
}

// FIXME: LangOpts is not properly saved because the LangOptions is not
// copyable! clang/lib/Frontend/SerializedDiagnosticPrinter.cpp does currently
// not serialize LangOpts either.
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
///
/// This allows the diagnostic to be emitted using the diagnostics engine, since
/// StandaloneDiagnostics themselfs cannot be emitted directly.
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
      OwnedFileMgr = llvm::makeIntrusiveRefCnt<FileManager>(std::move(Opts));
      OwnedSrcMgr =
          llvm::makeIntrusiveRefCnt<SourceManager>(Diags, *OwnedFileMgr);
    }
  }

  /// Emits \c StandaloneDiag using the associated DiagnosticsEngine.
  void Report(StandaloneDiagnostic &&StandaloneDiag) const {
    const auto StoredDiag = translateStandaloneDiag(
        getFileManager(), getSourceManager(), std::move(StandaloneDiag));
    Diags.getClient()->BeginSourceFile(StandaloneDiag.LangOpts, nullptr);
    Diags.Report(StoredDiag);
    Diags.getClient()->EndSourceFile();
  }

  /// Emits all diagnostics in \c StandaloneDiags using the associated
  /// DiagnosticsEngine.
  void Report(SmallVectorImpl<StandaloneDiagnostic> &&StandaloneDiags) const {
    for (auto &StandaloneDiag : StandaloneDiags)
      Report(std::move(StandaloneDiag));
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

  StandaloneDiagList takeDiagnostics() { return std::move(StandaloneDiags); }

private:
  StandaloneDiagList StandaloneDiags;
};
} // anonymous namespace

namespace {
/// The full dependencies for a single compilation input.
struct InputDependencies {
  /// The identifier of the C++20 module this translation unit exports.
  ///
  /// If the translation unit is not a module then \c ID.ModuleName is empty.
  ModuleID ID;

  /// Whether this is a "system" module.
  bool IsSystem;

  /// A collection of absolute paths to files that this translation unit
  /// directly depends on, not including transitive dependencies.
  std::vector<std::string> FileDeps;

  /// A list of modules this translation unit directly depends on, not including
  /// transitive dependencies.
  ///
  /// This may include modules with a different context hash when it can be
  /// determined that the differences are benign for this compilation.
  std::vector<ModuleID> ClangModuleDeps;

  /// A list of the C++20 named modules this translation unit depends on.
  std::vector<std::string> NamedModuleDeps;

  /// The compiler invocation with modifications to properly import all Clang
  /// module dependencies. Does not include argv[0].
  std::vector<std::string> BuildArgs;
};

/// The full dependencies for each compilation input and for all discovered
/// Clang modules.
struct DependencyScanResult {
  /// The full dependencies for each compilation input, in the same order as the
  /// inputs.
  ///
  /// System modules inputs that are not imported are represented as
  /// std::nullopt.
  llvm::SmallVector<std::optional<InputDependencies>> InputDeps;

  /// The full Clang module dependenies for this compilation.
  SmallVector<ModuleDeps, 0> ClangModuleDeps;
};

/// Merges and deterministically orders scan results from multiple threads
/// into a single DependencyScanResult.
class ScanResultCollector {
public:
  explicit ScanResultCollector(size_t NumInputs)
      : InputDeps(NumInputs), ClangModuleGraphs(NumInputs) {}

  /// Adds the dependency scan result for the input at \c InputIndex.
  ///
  /// Thread safe, given that each index is written to exactly once.
  void handleTUResult(TranslationUnitDeps &&TUDeps, bool IsSystem,
                      size_t InputIndex);

  /// Finalizes and takes the aggregated results.
  ///
  /// Not thread-safe.
  DependencyScanResult takeScanResults();

private:
  SmallVector<std::optional<InputDependencies>, 0> InputDeps;
  llvm::SmallVector<std::vector<ModuleDeps>, 0> ClangModuleGraphs;
};
} // anonymous namespace

void ScanResultCollector::handleTUResult(TranslationUnitDeps &&TUDeps,
                                         bool IsSystem, size_t InputIndex) {
  assert(!InputDeps[InputIndex].has_value() &&
         "Each slot should be written to at most once.");
  InputDeps[InputIndex].emplace();
  auto &NewInputDep = InputDeps[InputIndex].value();
  NewInputDep.ID = std::move(TUDeps.ID);
  NewInputDep.IsSystem = IsSystem;
  NewInputDep.FileDeps = std::move(TUDeps.FileDeps);
  NewInputDep.NamedModuleDeps = std::move(TUDeps.NamedModuleDeps);
  NewInputDep.ClangModuleDeps = std::move(TUDeps.ClangModuleDeps);
  assert(TUDeps.Commands.size() == 1 && "Expected exactly one command");
  NewInputDep.BuildArgs = TUDeps.Commands.front().Arguments;

  assert(ClangModuleGraphs[InputIndex].empty() &&
         "Each slot should be written to at most once.");
  ClangModuleGraphs[InputIndex] = std::move(TUDeps.ModuleGraph);
}

DependencyScanResult ScanResultCollector::takeScanResults() {
  DependencyScanResult Out;

  // Record each module only once, from its first importing input.
  // This keeps the output deterministic.
  llvm::DenseSet<ModuleID> AlreadySeen;
  for (auto &ModuleGraph : ClangModuleGraphs) {
    for (auto &MD : ModuleGraph) {
      auto [It, Inserted] = AlreadySeen.insert(MD.ID);
      if (!Inserted)
        continue;
      Out.ClangModuleDeps.push_back(std::move(MD));
    }
  }

  Out.InputDeps = std::move(InputDeps);
  return Out;
}

namespace {
/// Pool of reusable dependency scanning workers and their contexts, with RAII
/// acquire/release semantics.
class ScanningWorkerPool {
public:
  ScanningWorkerPool(size_t NumWorkers, DependencyScanningService &S,
                     llvm::vfs::FileSystem &FS);

  /// Acquires a unique pointer to a dependency scanning worker and its
  /// context.
  ///
  /// The worker bundle automatically released back to the pool when the
  /// pointer is destroyed. The pool has to outlive the leased worker bundle.
  [[nodiscard]] auto scopedAcquire();

private:
  /// Releases the worker bundle at \c Index back into the pool.
  void release(size_t Index);

  /// A scanning worker with its associated context.
  struct WorkerBundle {
    WorkerBundle(DependencyScanningService &S, llvm::vfs::FileSystem &FS)
        : Worker(S, &FS) {}

    DependencyScanningWorker Worker;
    llvm::DenseSet<ModuleID> SeenModules;
  };

  std::mutex Lock;
  std::condition_variable CV;
  SmallVector<size_t> AvailableSlots;
  SmallVector<WorkerBundle> Slots;
};
} // anonymous namespace

ScanningWorkerPool::ScanningWorkerPool(size_t NumWorkers,
                                       DependencyScanningService &S,
                                       llvm::vfs::FileSystem &FS) {
  Slots.reserve(NumWorkers);
  for (size_t I = 0; I < NumWorkers; ++I)
    Slots.emplace_back(S, FS);

  AvailableSlots.resize(NumWorkers);
  std::iota(AvailableSlots.begin(), AvailableSlots.end(), 0);
}

[[nodiscard]] auto ScanningWorkerPool::scopedAcquire() {
  std::unique_lock<std::mutex> UL(Lock);
  CV.wait(UL, [&] { return !AvailableSlots.empty(); });
  const auto Index = AvailableSlots.pop_back_val();
  auto ReleaseHandle = [this, Index](WorkerBundle *) { release(Index); };
  return std::unique_ptr<WorkerBundle, decltype(ReleaseHandle)>(&Slots[Index],
                                                                ReleaseHandle);
}

void ScanningWorkerPool::release(size_t Index) {
  {
    std::scoped_lock<std::mutex> SL(Lock);
    AvailableSlots.push_back(Index);
  }
  CV.notify_one();
}

namespace {
/// Thread-safe registry of system modules.
///
/// Records which system modules have been scheduled, and provides lookup for
/// input indices of system modules that have not yet been seen.
class SystemInputRegistry {
public:
  SystemInputRegistry(size_t FirstSystemInputIndex,
                      const StdModuleManifest &Manifest);

  /// Returns the indices of systems inputs that have not yet been seen.
  SmallVector<size_t> getNewSystemInputs(ArrayRef<std::string> NamedDeps);

private:
  size_t FirstSystemInputIndex;
  std::mutex Lock;
  llvm::DenseMap<llvm::StringRef, size_t> NameToManifestIndex;
  llvm::SmallBitVector ScheduledSystemModules;
};
} // anonymous namespace

SystemInputRegistry::SystemInputRegistry(size_t FirstSystemInputIndex,
                                         const StdModuleManifest &Manifest)
    : FirstSystemInputIndex(FirstSystemInputIndex),
      ScheduledSystemModules(Manifest.ModuleEntries.size(), false) {
  // Build the mapping from module names to their manifest index.
  NameToManifestIndex.reserve(Manifest.ModuleEntries.size());
  for (const auto &[Index, M] : llvm::enumerate(Manifest.ModuleEntries))
    NameToManifestIndex.try_emplace(M.LogicalName, Index);
}

SmallVector<size_t>
SystemInputRegistry::getNewSystemInputs(ArrayRef<std::string> NamedDeps) {
  SmallVector<size_t, 8> ToSchedule;
  {
    std::scoped_lock<std::mutex> SL(Lock);
    for (const auto &ModuleName : NamedDeps) {
      const auto It = NameToManifestIndex.find(ModuleName);
      if (It == NameToManifestIndex.end())
        continue;
      const auto ManifestIndex = It->second;
      if (ScheduledSystemModules[ManifestIndex])
        continue;
      ScheduledSystemModules[ManifestIndex] = true;
      ToSchedule.push_back(FirstSystemInputIndex + ManifestIndex);
    }
  }
  return ToSchedule;
}

/// Construct a path for the explicitly built PCM.
static std::string constructPCMPath(ModuleID ID, StringRef OutputDir) {
  assert(!ID.ModuleName.empty() && !ID.ContextHash.empty() &&
         "Invalid ModuleID");
  SmallString<256> ExplicitPCMPath(OutputDir);
  llvm::sys::path::append(ExplicitPCMPath, ID.ContextHash,
                          ID.ModuleName + "-" + ID.ContextHash + ".pcm");
  return std::string(ExplicitPCMPath);
}

namespace {
/// A simple dependency action controller that only provides module lookup for
/// Clang modules.
class ModuleLookupController : public DependencyActionController {
public:
  ModuleLookupController(StringRef OutputDir) : OutputDir(OutputDir) {}

  std::string lookupModuleOutput(const ModuleDeps &MD,
                                 ModuleOutputKind Kind) override {
    if (Kind == tooling::dependencies::ModuleOutputKind::ModuleFile)
      return constructPCMPath(MD.ID, OutputDir);

    // Driver command lines that trigger lookups for unsupported
    // ModuleOutputKinds are not supported by the modules driver. Those command
    // lines should probably be adjusted or rejected in Driver::handleArguments
    // or Driver::HandleImmediateArgs.
    llvm::reportFatalInternalError(
        "call to lookupModuleOutput with unexpected ModuleOutputKind");
  }

private:
  std::string OutputDir;
};
} // anonymous namespace

/// Constructs the full -cc1 command line, including executable, for the given
/// driver \c Job.
static std::vector<std::string> buildCC1CommandLine(const Command &Job) {
  assert(StringRef(Job.getCreator().getName()) == "clang");
  const auto &JobArgs = Job.getArguments();
  std::vector<std::string> CC1CommandLine;
  CC1CommandLine.reserve(JobArgs.size() + 1);
  CC1CommandLine.emplace_back(Job.getExecutable());
  for (const auto &Arg : JobArgs)
    CC1CommandLine.emplace_back(Arg);
  return CC1CommandLine;
}

/// Performs a full dependency scan of the given compilation inputs.
///
/// Diagnostics are emitted through the driver's diagnostic engine.
///
/// \returns A \c DependencyScanResult on success, or \c std::nullopt on
/// failure. The returned \c DependencyScanResult is deterministic for the given
/// compilation inputs.
static std::optional<DependencyScanResult> scanDependencies(
    const JobVector &ScanInputJobs, const StdModuleManifest &Manifest,
    size_t FirstSystemInputIndex, Compilation &C, StringRef ModuleDir) {
  llvm::PrettyStackTraceString CrashInfo("Performing module dependency scan.");

  DependencyScanningService ScanningService(
      ScanningMode::DependencyDirectivesScan, ScanningOutputFormat::Full);

  const auto NumInputs = ScanInputJobs.size();

  // TODO: Benchmark: Determine the optimal number of worker threads for a given
  // number of inputs. How many inputs are required for multi-threading to be
  // beneficial? How many inputs should each thread scan?
  std::unique_ptr<llvm::ThreadPoolInterface> ThreadPool;
  size_t WorkerCount;

#if LLVM_ENABLE_THREADS
  const bool HasSystemInputs = !Manifest.ModuleEntries.empty();

  if (NumInputs <= 2) {
    auto S = llvm::optimal_concurrency(1);
    ThreadPool = std::make_unique<llvm::SingleThreadExecutor>(std::move(S));
    WorkerCount = 1;
  } else {
    auto S = llvm::optimal_concurrency(NumInputs -
                                       static_cast<size_t>(HasSystemInputs));
    ThreadPool = std::make_unique<llvm::DefaultThreadPool>(std::move(S));
    const size_t MaxConcurrency = ThreadPool->getMaxConcurrency();
    WorkerCount = std::min(
        MaxConcurrency,
        NumInputs - (HasSystemInputs && NumInputs < MaxConcurrency ? 1 : 0));
  }
#else
  ThreadPool = std::make_unique<llvm::SingleThreadExecutor>();
  WorkerCount = 1;
#endif

  ScanningWorkerPool WorkerPool(WorkerCount, ScanningService,
                                C.getDriver().getVFS());

  SystemInputRegistry SysInputRegistry(FirstSystemInputIndex, Manifest);
  ModuleLookupController LookupController(ModuleDir);
  ScanResultCollector ResultCollector(NumInputs);
  SmallVector<StandaloneDiagList, 0> DiagLists(NumInputs);
  std::atomic<bool> HasError = false;

  std::function<void(size_t, bool)> ScanFn;
  ScanFn = [&](size_t InputIndex, bool IsSystem) {
    auto CC1CommandLine = buildCC1CommandLine(*ScanInputJobs[InputIndex]);
    StandaloneDiagCollector DiagConsumer;

    std::optional<TranslationUnitDeps> TUDeps;
    {
      auto WorkerHandle = WorkerPool.scopedAcquire();
      FullDependencyConsumer FullDepsConsumer(WorkerHandle->SeenModules);
      if (WorkerHandle->Worker.computeDependencies(
              ".", CC1CommandLine, FullDepsConsumer, LookupController,
              DiagConsumer))
        TUDeps = FullDepsConsumer.takeTranslationUnitDeps();
    }

    // Always capture diagnostics, since even successful scans may produce
    // warnings or notes.
    DiagLists[InputIndex] = DiagConsumer.takeDiagnostics();

    if (!TUDeps) {
      HasError.store(true, std::memory_order_relaxed);
      return;
    }

    // Enqueue scans for system modules newly required by this TU.
    for (auto SysInputIndex :
         SysInputRegistry.getNewSystemInputs(TUDeps->NamedModuleDeps))
      ThreadPool->async(
          [&ScanFn, SysInputIndex]() { ScanFn(SysInputIndex, true); });

    ResultCollector.handleTUResult(std::move(*TUDeps), IsSystem, InputIndex);
  };

  // Initiate the dependency scan with all user inputs.
  for (size_t I = 0; I < FirstSystemInputIndex; ++I)
    ThreadPool->async([&ScanFn, I]() { ScanFn(I, /*IsSystem*/ false); });
  ThreadPool->wait();

  // Report the diagnostics for each dependency scan.
  StandaloneDiagReporter DiagReporter(C.getDriver().getDiags());
  for (auto &DiagsList : DiagLists)
    DiagReporter.Report(std::move(DiagsList));

  if (HasError)
    return std::nullopt;

  return ResultCollector.takeScanResults();
}

namespace {
class MDGNode;
class MDGEdge;
using MDGNodeBase = llvm::DGNode<MDGNode, MDGEdge>;
using MDGEdgeBase = llvm::DGEdge<MDGNode, MDGEdge>;
using ModuleDependencyGraphBase = llvm::DirectedGraph<MDGNode, MDGEdge>;

/// Abstract base class for all node kinds in the module dependency graph.
class MDGNode : public MDGNodeBase {
public:
  enum class NodeKind {
    Root,
    ClangModule,
    NamedModule,
    NonModule,
  };

  explicit MDGNode(NodeKind Kind) : Kind(Kind) {}
  virtual ~MDGNode() = 0;

  /// Returns this node's kind.
  NodeKind getKind() const { return Kind; }

  /// Returns this node's Clang module dependencies.
  virtual ArrayRef<ModuleID> getClangDeps() const { return {}; }

  /// Returns this node's C++ named module dependencies.
  virtual ArrayRef<std::string> getNamedDeps() const { return {}; }

private:
  const NodeKind Kind;
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
  /// Define classof to be able to use isa<>, cast<>, dyn_cast<>, etc.
  static bool classof(const MDGNode *N) {
    return N->getKind() == NodeKind::Root;
  }

private:
  // Only allow the graph itself to creates its own single root node.
  friend class ModuleDependencyGraph;

  RootMDGNode() : MDGNode(NodeKind::Root) {}
};

class TranslationUnitMDGNode : public MDGNode {
protected:
  TranslationUnitMDGNode(NodeKind K, const InputDependencies &D)
      : MDGNode(K), InputDeps(D) {}

public:
  StringRef getFilename() const {
    assert(!InputDeps.FileDeps.empty() &&
           "Expected input itself to be the first file dependency.");
    return InputDeps.FileDeps.front();
  }

  ArrayRef<ModuleID> getClangDeps() const final {
    return InputDeps.ClangModuleDeps;
  }

  ArrayRef<std::string> getNamedDeps() const final {
    return InputDeps.NamedModuleDeps;
  }

  const InputDependencies &InputDeps;
};

/// Subclass of MDGNode representing a translation unit that provides no
/// module of any kind.
class NonModuleMDGNode final : public TranslationUnitMDGNode {
public:
  explicit NonModuleMDGNode(const InputDependencies &InputDeps)
      : TranslationUnitMDGNode(NodeKind::NonModule, InputDeps) {}

  /// Define classof to be able to use isa<>, cast<>, dyn_cast<>, etc.
  static bool classof(const MDGNode *N) {
    return N->getKind() == NodeKind::NonModule;
  }
};

/// Subclass of MDGNode representing a translation unit that provides a C++20
/// named module interface unit.
class CXXNamedModuleMDGNode final : public TranslationUnitMDGNode {
public:
  explicit CXXNamedModuleMDGNode(const InputDependencies &InputDeps)
      : TranslationUnitMDGNode(NodeKind::NamedModule, InputDeps) {}

  const ModuleID &getModuleID() const { return InputDeps.ID; };

  /// Define classof to be able to use isa<>, cast<>, dyn_cast<>, etc.
  static bool classof(const MDGNode *N) {
    return N->getKind() == NodeKind::NamedModule;
  }
};

/// Subclass of MDGNode representing a Clang module unit.
class ClangModuleMDGNode final : public MDGNode {
public:
  explicit ClangModuleMDGNode(const ModuleDeps &ClangModuleDeps)
      : MDGNode(NodeKind::ClangModule), ModuleDeps(ClangModuleDeps) {}

  /// Define classof to be able to use isa<>, cast<>, dyn_cast<>, etc.
  static bool classof(const MDGNode *N) {
    return N->getKind() == NodeKind::ClangModule;
  }
  const ModuleID &getModuleID() const { return ModuleDeps.ID; }

  ArrayRef<ModuleID> getClangDeps() const override {
    return ModuleDeps.ClangModuleDeps;
  }

  const ModuleDeps &ModuleDeps;
};

/// Represents an import relation in the module dependency graph, directed
/// from the imported module to the importer.
class MDGEdge : public MDGEdgeBase {
public:
  explicit MDGEdge(MDGNode &N) : MDGEdgeBase(N) {}
  MDGEdge() = delete;
};

/// A directed graph describing the full dependency relations of this
/// compilation.
///
/// The graph owns all of its nodes and edges.
/// The graph's root node is initialized on construction.
class ModuleDependencyGraph : public ModuleDependencyGraphBase {
public:
  ModuleDependencyGraph() {
    Root = new (Alloc.Allocate(sizeof(RootMDGNode), alignof(RootMDGNode)))
        RootMDGNode();
    addNode(*Root);
  }

  MDGNode *getRoot() { return Root; }
  const MDGNode *getRoot() const { return Root; }

  llvm::BumpPtrAllocator &getAllocator() { return Alloc; }

private:
  llvm::BumpPtrAllocator Alloc;
  RootMDGNode *Root = nullptr;
};
} // anonymous namespace

/// Allocation helper using the graph's allocator.
template <typename MDGComponent, typename... Args>
static MDGComponent *makeWithAlloc(llvm::BumpPtrAllocator &Alloc,
                                   Args &&...args) {
  return new (Alloc.Allocate(sizeof(MDGComponent), alignof(MDGComponent)))
      MDGComponent(std::forward<Args>(args)...);
}

/// Builds all Clang module nodes from \c ClangModuleDeps for \c Graph and
/// registers them in \c ClangModuleMap.
static void buildClangModuleNodes(
    ModuleDependencyGraph &Graph, ArrayRef<ModuleDeps> ClangModuleDeps,
    llvm::DenseMap<ModuleID, ClangModuleMDGNode *> &ClangModuleMap) {
  auto &Alloc = Graph.getAllocator();
  for (auto &M : ClangModuleDeps) {
    auto *Node = makeWithAlloc<ClangModuleMDGNode>(Alloc, M);
    Graph.addNode(*Node);
    const auto [It, Inserted] = ClangModuleMap.try_emplace(M.ID, Node);
    assert(Inserted && "Duplicate Clang module in scan result.");
  }
}

/// Builds translation unit nodes from \c InputDeps for \c Graph and registers
/// all named module nodes in \c NamedModuleMap.
///
/// \returns true on success, if no duplicate modules are detected.
static bool buildTranslationUnitNodes(
    ModuleDependencyGraph &Graph,
    ArrayRef<std::optional<InputDependencies>> InputDeps,
    llvm::DenseMap<StringRef, CXXNamedModuleMDGNode *> &NamedModuleMap,
    DiagnosticsEngine &Diags) {
  auto &Alloc = Graph.getAllocator();
  bool HasDuplicate = false;

  for (auto &Dep : InputDeps) {
    if (!Dep)
      continue;

    MDGNode *Node = nullptr;
    if (Dep->ID.ModuleName.empty()) {
      // If there is no module name, this is a regular non-module TU.
      Node = makeWithAlloc<NonModuleMDGNode>(Alloc, *Dep);
    } else {
      // There is a module name; this is a named module interface unit.
      Node = makeWithAlloc<CXXNamedModuleMDGNode>(Alloc, *Dep);
      const auto [It, Inserted] = NamedModuleMap.try_emplace(
          Dep->ID.ModuleName, static_cast<CXXNamedModuleMDGNode *>(Node));
      if (!Inserted) {
        // We have multiple source files which define the same module.
        Diags.Report(diag::err_mod_graph_named_module_redefinition)
            << Dep->ID.ModuleName << It->second->InputDeps.FileDeps.front()
            << Dep->FileDeps.front();
        HasDuplicate = true;
      }
    }

    Graph.addNode(*Node);
  }

  return !HasDuplicate;
}

/// Adds edges from \c Importer to nodes referenced in \c Map for each
/// dependency in \c Dependencies.
template <typename DepTy, typename MapTy>
static void addNodeEdges(MDGNode &Importer, ArrayRef<DepTy> Dependencies,
                         MapTy &Map, llvm::BumpPtrAllocator &Alloc) {
  for (const auto &Dep : Dependencies) {
    if (auto It = Map.find(Dep); It != Map.end()) {
      // TODO: For imports to unknown modules, check if a prebuilt module is
      // provided via -fmodule-file.
      // FIXME: TranslationUnitDeps::PrebuiltModuleDeps does currently not
      // provide this information for named modules provided via
      // -fmodule-file.
      auto *Edge = makeWithAlloc<MDGEdge>(Alloc, *It->second);
      Importer.addEdge(*Edge);
    }
  }
}

/// Connects all nodes in \c Graph to their dependencies and links the root
/// node to nodes with otherwise no incoming edges.
static void connectGraphNodes(
    ModuleDependencyGraph &Graph,
    const llvm::DenseMap<ModuleID, ClangModuleMDGNode *> &ClangModuleMap,
    const llvm::DenseMap<StringRef, CXXNamedModuleMDGNode *> &NamedModuleMap) {
  auto &Alloc = Graph.getAllocator();

  for (auto *Node : Graph) {
    addNodeEdges(*Node, Node->getClangDeps(), ClangModuleMap, Alloc);
    addNodeEdges(*Node, Node->getNamedDeps(), NamedModuleMap, Alloc);
  }

  for (auto *Node : Graph) {
    if (Node->getEdges().empty()) {
      auto *Edge = makeWithAlloc<MDGEdge>(Alloc, *Node);
      Graph.getRoot()->addEdge(*Edge);
    }
  }
}

/// Construct a module dependency graph from the scan result.
static std::optional<ModuleDependencyGraph>
buildModuleDependencyGraph(DependencyScanResult &ScanResult,
                           DiagnosticsEngine &Diags) {
  ModuleDependencyGraph Graph;

  // Lookup maps, used for connecting the nodes.
  llvm::DenseMap<ModuleID, ClangModuleMDGNode *> ClangModuleMap;
  llvm::DenseMap<StringRef, CXXNamedModuleMDGNode *> NamedModuleMap;

  buildClangModuleNodes(Graph, ScanResult.ClangModuleDeps, ClangModuleMap);
  if (!buildTranslationUnitNodes(Graph, ScanResult.InputDeps, NamedModuleMap,
                                 Diags))
    return std::nullopt;
  connectGraphNodes(Graph, ClangModuleMap, NamedModuleMap);

  return Graph;
}
} // namespace clang::driver::modules

namespace llvm {
namespace cdm = ::clang::driver::modules;

/// Non-const versions of the GraphTraits specializations for ModuleDepGraph.
template <> struct GraphTraits<cdm::MDGNode *> {
  using NodeRef = cdm::MDGNode *;

  static NodeRef MDGGetTargetNode(cdm::MDGEdgeBase *E) {
    return &E->getTargetNode();
  }

  using ChildIteratorType =
      mapped_iterator<cdm::MDGNode::iterator, decltype(&MDGGetTargetNode)>;
  using ChildEdgeIteratorType = cdm::MDGNode::iterator;

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
struct GraphTraits<cdm::ModuleDependencyGraph *> : GraphTraits<cdm::MDGNode *> {
  using GraphRef = cdm::ModuleDependencyGraph *;
  using NodeRef = cdm::MDGNode *;

  using nodes_iterator = cdm::ModuleDependencyGraph::iterator;

  static NodeRef getEntryNode(GraphRef G) { return G->getRoot(); }

  static nodes_iterator nodes_begin(GraphRef G) { return G->begin(); }

  static nodes_iterator nodes_end(GraphRef G) { return G->end(); }
};

/// Const versions of the GraphTraits specializations for ModuleDepGraph.
template <> struct GraphTraits<const cdm::MDGNode *> {
  using NodeRef = const cdm::MDGNode *;

  static NodeRef MDGGetTargetNode(const cdm::MDGEdgeBase *E) {
    return &E->getTargetNode();
  }

  using ChildIteratorType = mapped_iterator<cdm::MDGNode::const_iterator,
                                            decltype(&MDGGetTargetNode)>;
  using ChildEdgeIteratorType = cdm::MDGNode::const_iterator;

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
struct GraphTraits<const cdm::ModuleDependencyGraph *>
    : GraphTraits<const cdm::MDGNode *> {
  using GraphRef = const cdm::ModuleDependencyGraph *;
  using NodeRef = const cdm::MDGNode *;

  using nodes_iterator = cdm::ModuleDependencyGraph::const_iterator;

  static NodeRef getEntryNode(GraphRef G) { return G->getRoot(); }

  static nodes_iterator nodes_begin(GraphRef G) { return G->begin(); }

  static nodes_iterator nodes_end(GraphRef G) { return G->end(); }
};

template <>
struct DOTGraphTraits<const cdm::ModuleDependencyGraph *>
    : DefaultDOTGraphTraits {
  explicit DOTGraphTraits(bool IsSimple = false)
      : DefaultDOTGraphTraits(IsSimple) {}

  static std::string getGraphName(const cdm::ModuleDependencyGraph *) {
    return "Module Dependency Graph";
  }

  static std::string getGraphProperties(const cdm::ModuleDependencyGraph *) {
    return "\tnode [colorscheme=pastel13,style=filled,shape=Mrecord];\n\tedge "
           "[dir=\"back\"];\n";
  }

  static bool isNodeHidden(const cdm::MDGNode *N,
                           const cdm::ModuleDependencyGraph *G) {
    assert(N && "Node must not be null");
    return isa<cdm::RootMDGNode>(N);
  }

  static std::string getNodeIdentifier(const cdm::MDGNode *N,
                                       const cdm::ModuleDependencyGraph *) {
    using namespace cdm;

    assert(N && "Node must not be null");
    return llvm::TypeSwitch<const MDGNode *, std::string>(N)
        .Case<ClangModuleMDGNode>([](const ClangModuleMDGNode *Node) {
          const auto &ID = Node->getModuleID();
          return (Twine(ID.ModuleName) + ":" + ID.ContextHash).str();
        })
        .Case<CXXNamedModuleMDGNode>([](const CXXNamedModuleMDGNode *Node) {
          return Node->getModuleID().ModuleName;
        })
        .Case<NonModuleMDGNode>(
            [](const NonModuleMDGNode *Node) { return Node->getFilename(); });
  }

  static std::string getNodeLabel(const cdm::MDGNode *N,
                                  const cdm::ModuleDependencyGraph *) {
    using namespace cdm;

    assert(N && "Node must not be null");
    SmallString<128> Buf;
    raw_svector_ostream OS(Buf);

    auto PrintModuleKind = [&](StringRef Kind) { OS << "Kind: " << Kind; };
    auto PrintModuleName = [&](StringRef Name) {
      OS << "Module name: " << Name;
    };
    auto PrintFilename = [&](StringRef Filename) {
      OS << "Filename: " << Filename;
    };
    auto PrintInputOrigin = [&](bool IsSystem) {
      OS << "Input origin: " << (IsSystem ? "System" : "User");
    };
    auto PrintContextHash = [&](StringRef Hash) { OS << "Hash: " << Hash; };

    llvm::TypeSwitch<const MDGNode *>(N)
        .Case<ClangModuleMDGNode>([&](const ClangModuleMDGNode *Node) {
          PrintModuleKind("Clang module");
          OS << " \\| ";
          PrintModuleName(Node->getModuleID().ModuleName);
          OS << " \\| ";
          OS << "Modulemap file: " << Node->ModuleDeps.ClangModuleMapFile;
          OS << " \\| ";
          PrintInputOrigin(Node->ModuleDeps.IsSystem);
          OS << " \\| ";
          PrintContextHash(Node->getModuleID().ContextHash);
        })
        .Case<CXXNamedModuleMDGNode>([&](const CXXNamedModuleMDGNode *Node) {
          PrintModuleKind("C++ named module");
          OS << " \\| ";
          PrintModuleName(Node->getModuleID().ModuleName);
          OS << " \\| ";
          PrintFilename(Node->getFilename());
          OS << " \\| ";
          PrintInputOrigin(Node->InputDeps.IsSystem);
          OS << " \\| ";
          PrintContextHash(Node->getModuleID().ContextHash);
        })
        .Case<NonModuleMDGNode>([&](const NonModuleMDGNode *Node) {
          PrintModuleKind("Non-module");
          OS << " \\| ";
          PrintFilename(Node->getFilename());
        })
        .Default([](const MDGNode *) {
          llvm_unreachable("Unhandled MDGNode kind in getNodeLabel!");
        });

    return std::string(OS.str());
  }

  static std::string getNodeAttributes(const cdm::MDGNode *N,
                                       const cdm::ModuleDependencyGraph *) {
    using namespace cdm;

    assert(N && "Node must not be null");
    return llvm::TypeSwitch<const MDGNode *, std::string>(N)
        .Case<ClangModuleMDGNode>(
            [&](const ClangModuleMDGNode *) { return "fillcolor=1"; })
        .Case<CXXNamedModuleMDGNode>(
            [&](const CXXNamedModuleMDGNode *) { return "fillcolor=2"; })
        .Case<NonModuleMDGNode>(
            [&](const NonModuleMDGNode *) { return "fillcolor=3"; });
  }
};

/// GraphWriter specialization for ModuleDepGraph.
///
/// Uses human-readable identifiers instead of raw pointers and separates node
/// definitions from import edges for a more remark-friendly output.
template <>
class GraphWriter<const cdm::ModuleDependencyGraph *>
    : public GraphWriterBase<const cdm::ModuleDependencyGraph *,
                             GraphWriter<const cdm::ModuleDependencyGraph *>> {
public:
  using GraphType = const cdm::ModuleDependencyGraph *;
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

void llvm::GraphWriter<const cdm::ModuleDependencyGraph *>::writeNodes() {
  auto IsNodeVisible = [&](NodeRef N) { return !DTraits.isNodeHidden(N, G); };
  const auto VisibleNodeRange = make_filter_range(nodes(G), IsNodeVisible);
  const SmallVector<NodeRef, 0> VisibleNodes(VisibleNodeRange);

  writeNodeDefinitions(VisibleNodes);
  writeNodeRelations(VisibleNodes);
}

void llvm::GraphWriter<const cdm::ModuleDependencyGraph
                           *>::writeNodeDefinitions(ArrayRef<NodeRef> Nodes) {
  for (const auto &Node : Nodes) {
    const auto NodeID = DTraits.getNodeIdentifier(Node, G);
    const auto NodeLabel = DTraits.getNodeLabel(Node, G);
    O << "\t\"" << DOT::EscapeString(NodeID) << "\" [ "
      << DTraits.getNodeAttributes(Node, G) << ",label=\"{ "
      << DOT::EscapeString(NodeLabel) << " }\"];\n";
    NodeIDMap.try_emplace(Node, std::move(NodeID));
  }
  O << "\n";
}

void llvm::GraphWriter<const cdm::ModuleDependencyGraph *>::writeNodeRelations(
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

namespace clang::driver::modules {
/// Returns true if the driver \c Job is eligible as a dependency scan
/// input.
///
/// A job is eligible if it is a clang \c -cc1 invocation and all its inputs
/// are source files.
static bool isJobForDependencyScan(const Command &Job) {
  if (StringRef(Job.getCreator().getName()) != "clang")
    return false;
  auto IsSrcInput = [](const InputInfo &II) -> bool {
    return types::isSrcFile(II.getType());
  };
  return llvm::all_of(Job.getInputInfos(), IsSrcInput);
}

/// Splits driver jobs into those eligible for dependency scanning and the rest.
///
/// Non-eligible jobs are either not \c -cc1 jobs or depend on outputs from
/// eligible jobs. The original order of jobs is preserved.
///
/// \returns the pair: (dependency scan input jobs, rest jobs).
static std::pair<JobVector, JobVector>
splitCommandsByScanEligibility(JobVector &&Jobs) {
  std::pair<JobVector, JobVector> Result;
  auto &[ScanInputJobs, OtherJobs] = Result;
  for (auto &Job : Jobs) {
    if (isJobForDependencyScan(*Job))
      ScanInputJobs.push_back(std::move(Job));
    else
      OtherJobs.push_back(std::move(Job));
  }
  return Result;
}

/// Adds each system include directory in \p SystemIncludeDirs to \p Job's
/// arguments.
static void
addSystemIncludeDirsFromManifest(Compilation &C, Command &Job,
                                 ArrayRef<std::string> SystemIncludeDirs) {
  const auto &TC = Job.getCreator().getToolChain();
  const auto &TCArgs = C.getArgsForToolChain(
      &TC, /*BoundArch*/ "", Job.getSource().getOffloadingDeviceKind());
  auto &CC1Args = Job.getArguments();

  for (const auto &IncludeDir : SystemIncludeDirs)
    TC.addSystemInclude(TCArgs, CC1Args, IncludeDir);
}

/// Applies local module arguments from the \c Manifest to the corresponding
/// system input jobs in \c ScanJobs.
static void applyLocalArgsFromManifest(Compilation &C, JobVector &ScanJobs,
                                       size_t FirstSystemInputIndex,
                                       const StdModuleManifest &Manifest) {
  auto SystemInputJobs = ArrayRef(ScanJobs).slice(FirstSystemInputIndex);
  for (auto &&[Job, ManifestEntry] :
       llvm::zip_equal(SystemInputJobs, Manifest.ModuleEntries)) {
    const auto &LocalArgs = ManifestEntry.LocalArgs;
    if (LocalArgs)
      addSystemIncludeDirsFromManifest(C, *Job, LocalArgs->SystemIncludeDirs);
  }
}

/// Computes the -fmodule-cache-path for this compilation.
static SmallString<128> getModuleCachePath(DerivedArgList &Args) {
  SmallString<128> Path;
  if (const auto &A = Args.getLastArg(options::OPT_fmodules_cache_path))
    Path = A->getValue();
  else
    driver::Driver::getDefaultModuleCachePath(Path);
  return Path;
}

void planDriverManagedModuleCompilation(Compilation &C,
                                        const StdModuleManifest &Manifest) {
  llvm::PrettyStackTraceString CrashInfo("Planning modules build.");
  auto &Diags = C.getDriver().getDiags();

  auto [ScanInputJobs, OtherJobs] =
      splitCommandsByScanEligibility(C.getJobs().takeJobs());

  const size_t SystemInputCount = Manifest.ModuleEntries.size();
  const size_t FirstSystemInputIndex = ScanInputJobs.size() - SystemInputCount;

#ifndef NDEBUG
  // System module inputs are appended to the end of the compilation input list,
  // in the same order as they appear in the standard module manifest.
  // We assume these manifest-provided inputs are appended last.
  // If this assumption ever breaks, explicitly partition the jobs by matching
  // their inputs against the manifest paths.
  auto SysInputJobsForScan =
      ArrayRef(ScanInputJobs).take_back(SystemInputCount);
  for (const auto &[Job, ManifestEntry] :
       llvm::zip_equal(SysInputJobsForScan, Manifest.ModuleEntries)) {
    const auto &InputInfos = Job->getInputInfos();
    assert(InputInfos.size() == 1 && "Expected exactly one dependency "
                                     "scanning input for each system module");
    assert(InputInfos.front().getFilename() == ManifestEntry.SourcePath &&
           "System module input order should match order in manifest!");
  }
#endif

  // Apply manifest-specified local arguments before scanning as they may affect
  // the scan results.
  applyLocalArgsFromManifest(C, ScanInputJobs, FirstSystemInputIndex, Manifest);

  auto ModuleCachePath = getModuleCachePath(C.getArgs());

  // Run the dependency scan.
  auto ScanResults = scanDependencies(
      ScanInputJobs, Manifest, FirstSystemInputIndex, C, ModuleCachePath);
  if (!ScanResults) {
    C.getDriver().getDiags().Report(diag::err_failed_dependency_scan);
    return;
  }

  // TODO: Remove driver jobs for module inputs generated from the standard
  // library manifest that are unused in this compilation (i.e., each job
  // corresponding to a std::nullopt in ScanResults).

  // TODO: Generate driver jobs for each Clang module and update the other
  // driver jobs with the scan-generated command lines.

  // Construct the module dependency graph.
  // TODO: Attach all jobs to the graph during construction.
  auto MaybeModuleGraph = buildModuleDependencyGraph(*ScanResults, Diags);
  if (!MaybeModuleGraph) {
    Diags.Report(diag::err_building_dependency_graph);
    return;
  }
  Diags.Report(diag::remark_printing_module_graph);
  if (!Diags.isLastDiagnosticIgnored())
    llvm::WriteGraph<const ModuleDependencyGraph *>(llvm::errs(),
                                                    &(*MaybeModuleGraph));

  // TODO: Check for cyclic dependencies in the module dependency graph.

  // TODO: Modify each driver job's command line to generate/pass-in the right
  // module files.

  // Merge the jobs back into the compilation's job list.
  for (auto &Job :
       llvm::concat<std::unique_ptr<Command>>(ScanInputJobs, OtherJobs))
    C.addCommand(std::move(Job));
}
} // namespace clang::driver::modules
