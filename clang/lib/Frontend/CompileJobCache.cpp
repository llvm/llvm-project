//===- CompileJobCache.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/CompileJobCache.h"
#include "CachedDiagnostics.h"
#include "clang/Basic/DiagnosticCAS.h"
#include "clang/Frontend/CASDependencyCollector.h"
#include "clang/Frontend/CompileJobCacheKey.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/CAS/ActionCache.h"
#include "llvm/CAS/CASOutputBackend.h"
#include "llvm/CASUtil/Utils.h"
#include "llvm/MCCAS/MCCASObjectV1.h"
#include "llvm/RemoteCachingService/Client.h"
#include "llvm/Support/FileOutputBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/PrefixMapper.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/ScopedDurationTimer.h"
#include "llvm/Support/VirtualOutputBackends.h"

using namespace clang;
using llvm::Error;

/// Represents a mechanism for storing and retrieving compilation artifacts.
/// It includes common functionality and extension points for specific backend
/// implementations.
class CompileJobCache::CachingOutputs {
public:
  using OutputKind = clang::cas::CompileJobCacheResult::OutputKind;

  CachingOutputs(CompilerInstance &Clang, StringRef Workingdir,
                 llvm::PrefixMapper Mapper, bool WriteOutputAsCASID,
                 bool UseCASBackend);
  virtual ~CachingOutputs() = default;

  /// \returns true if result was found and replayed, false otherwise.
  virtual Expected<bool>
  tryReplayCachedResult(const llvm::cas::CASID &ResultCacheKey,
                        const llvm::cas::CASID &CanonicalResultCacheKey) = 0;

  /// \returns true on failure, false on success.
  virtual bool prepareOutputCollection() = 0;

  void stopDiagnosticsCapture();

  /// This is for an output file that was written directly on the file system.
  /// It's a workaround until all compilation consumers adopt
  /// \c llvm::vfs::OutputBackend.
  virtual Error addNonVirtualOutputFile(StringRef FilePath) = 0;

  /// Finish writing outputs from a computed result, after a cache miss.
  /// If SkipCache is true, it should not insert the ResultCacheKey into
  /// Cache for future uses.
  virtual Error
  finishComputedResult(const llvm::cas::CASID &ResultCacheKey,
                       const llvm::cas::CASID &CanonicalResultCacheKey,
                       bool SkipCache) = 0;

protected:
  StringRef getPathForOutputKind(OutputKind Kind);

  bool prepareOutputCollectionCommon(
      IntrusiveRefCntPtr<llvm::vfs::OutputBackend> CacheOutputs);
  Error replayCachedDiagnostics(StringRef DiagsData);

  CompilerInstance &Clang;
  const llvm::PrefixMapper PrefixMapper;
  const bool WriteOutputAsCASID;
  const bool UseCASBackend;
  clang::cas::CompileJobCacheResult::Builder CachedResultBuilder;
  std::string OutputFile;
  std::string DependenciesFile;
  std::unique_ptr<clang::cas::CachingDiagnosticsProcessor> DiagProcessor;
};

namespace {

/// Store and retrieve compilation artifacts using \p llvm::cas::ObjectStore and
/// \p llvm::cas::ActionCache.
class ObjectStoreCachingOutputs : public CompileJobCache::CachingOutputs {
public:
  ObjectStoreCachingOutputs(CompilerInstance &Clang, StringRef WorkingDir,
                            llvm::PrefixMapper Mapper, bool WriteOutputAsCASID,
                            bool UseCASBackend,
                            std::optional<llvm::cas::CASID> &MCOutputID,
                            std::shared_ptr<llvm::cas::ObjectStore> DB,
                            std::shared_ptr<llvm::cas::ActionCache> Cache)
      : CachingOutputs(Clang, WorkingDir, std::move(Mapper), WriteOutputAsCASID,
                       UseCASBackend),
        ComputedJobNeedsReplay(WriteOutputAsCASID || UseCASBackend),
        MCOutputID(MCOutputID), CAS(std::move(DB)), Cache(std::move(Cache)) {
    if (CAS)
      CASOutputs = llvm::makeIntrusiveRefCnt<llvm::cas::CASOutputBackend>(*CAS);
  }

  Expected<std::optional<int>>
  replayCachedResult(const llvm::cas::CASID &ResultCacheKey,
                     clang::cas::CompileJobCacheResult &Result,
                     bool JustComputedResult);

private:
  Expected<bool> tryReplayCachedResult(
      const llvm::cas::CASID &ResultCacheKey,
      const llvm::cas::CASID &CanonicalResultCacheKey) override;

  bool prepareOutputCollection() override;

  Error addNonVirtualOutputFile(StringRef FilePath) override;

  Error finishComputedResult(const llvm::cas::CASID &ResultCacheKey,
                             const llvm::cas::CASID &CanonicalResultCacheKey,
                             bool SkipCache) override;

  Expected<llvm::cas::ObjectRef>
  writeOutputs(const llvm::cas::CASID &ResultCacheKey);

  /// Replay a cache hit.
  ///
  /// Return status if should exit immediately, otherwise std::nullopt.
  std::optional<int> replayCachedResult(const llvm::cas::CASID &ResultCacheKey,
                                        llvm::cas::ObjectRef ResultID,
                                        bool JustComputedResult);

  const bool ComputedJobNeedsReplay;
  std::optional<llvm::cas::CASID> &MCOutputID;
  std::shared_ptr<llvm::cas::ObjectStore> CAS;
  std::shared_ptr<llvm::cas::ActionCache> Cache;
  IntrusiveRefCntPtr<llvm::cas::CASOutputBackend> CASOutputs;
  std::optional<llvm::cas::ObjectRef> DependenciesOutput;
};

/// An \p OutputBackend that just records the list of output paths/names.
class CollectingOutputBackend : public llvm::vfs::ProxyOutputBackend {
  SmallVector<std::string> OutputNames;

public:
  CollectingOutputBackend()
      : llvm::vfs::ProxyOutputBackend(llvm::vfs::makeNullOutputBackend()) {}

  ArrayRef<std::string> getOutputs() const { return OutputNames; }

  /// Add an association of a "kind" string with a particular output path.
  /// When the output for \p Path is encountered it will be associated with
  /// the \p Kind string instead of its path.
  void addKindMap(StringRef Kind, StringRef Path) {
    KindMaps.push_back({Saver.save(Kind), Saver.save(Path)});
  }

  void addOutput(StringRef Path) {
    StringRef Name = tryRemapPath(Path);
    OutputNames.push_back(Name.str());
  }

private:
  llvm::BumpPtrAllocator Alloc;
  llvm::StringSaver Saver{Alloc};

  struct KindMap {
    StringRef Kind;
    StringRef Path;
  };
  SmallVector<KindMap> KindMaps;

  /// Returns the "kind" name for the path if one was added for it, otherwise
  /// returns the \p Path itself.
  StringRef tryRemapPath(StringRef Path) const {
    for (const KindMap &Map : KindMaps) {
      if (Map.Path == Path)
        return Map.Kind;
    }
    return Path;
  }

  Expected<std::unique_ptr<llvm::vfs::OutputFileImpl>>
  createFileImpl(StringRef Path,
                 std::optional<llvm::vfs::OutputConfig> Config) override {
    addOutput(Path);
    return ProxyOutputBackend::createFileImpl(Path, std::move(Config));
  }

  IntrusiveRefCntPtr<llvm::vfs::OutputBackend> cloneImpl() const override {
    return IntrusiveRefCntPtr<CollectingOutputBackend>(
        const_cast<CollectingOutputBackend *>(this));
  }
};

/// Store and retrieve compilation artifacts using \p llvm::cas::CASDBClient
/// and \p llvm::cas::KeyValueDBClient.
class RemoteCachingOutputs : public CompileJobCache::CachingOutputs {
public:
  RemoteCachingOutputs(CompilerInstance &Clang, StringRef WorkingDir,
                       llvm::PrefixMapper Mapper,
                       llvm::cas::remote::ClientServices Clients)
      : CachingOutputs(Clang, WorkingDir, std::move(Mapper),
                       /*WriteOutputAsCASID*/ false,
                       /*UseCASBackend*/ false) {
    RemoteKVClient = std::move(Clients.KVDB);
    RemoteCASClient = std::move(Clients.CASDB);
    CollectingOutputs = llvm::makeIntrusiveRefCnt<CollectingOutputBackend>();
  }

private:
  Expected<bool> tryReplayCachedResult(
      const llvm::cas::CASID &ResultCacheKey,
      const llvm::cas::CASID &CanonicalResultCacheKey) override;

  bool prepareOutputCollection() override;

  Error addNonVirtualOutputFile(StringRef FilePath) override;

  Error finishComputedResult(const llvm::cas::CASID &ResultCacheKey,
                             const llvm::cas::CASID &CanonicalResultCacheKey,
                             bool SkipCache) override;

  Expected<llvm::cas::remote::KeyValueDBClient::ValueTy>
  saveOutputs(const llvm::cas::CASID &ResultCacheKey);

  Expected<bool> replayCachedResult(
      const llvm::cas::CASID &ResultCacheKey,
      const llvm::cas::remote::KeyValueDBClient::ValueTy &CompResult);

  std::string getPrintableRemoteID(StringRef RemoteCASIDBytes);

  void tryReleaseLLBuildExecutionLane();

  static StringRef getOutputKindName(OutputKind Kind);
  /// \returns \p std::nullopt if \p Name doesn't match one of the output kind names.
  static std::optional<OutputKind> getOutputKindForName(StringRef Name);

  std::unique_ptr<llvm::cas::remote::KeyValueDBClient> RemoteKVClient;
  std::unique_ptr<llvm::cas::remote::CASDBClient> RemoteCASClient;
  IntrusiveRefCntPtr<CollectingOutputBackend> CollectingOutputs;
  bool TriedReleaseLLBuildExecutionLane = false;
};

} // anonymous namespace

StringRef
CompileJobCache::CachingOutputs::getPathForOutputKind(OutputKind Kind) {
  switch (Kind) {
  case OutputKind::MainOutput:
    return OutputFile;
  case OutputKind::Dependencies:
    return DependenciesFile;
  default:
    return "";
  }
}

static std::string fixupRelativePath(const std::string &Path, FileManager &FM,
                                     StringRef WorkingDir) {
  if (llvm::sys::path::is_absolute(Path) || Path.empty() || Path == "-")
    return Path;

  // Apply -working-dir compiler option.
  // FIXME: this needs to stay in sync with createOutputFileImpl. Ideally, clang
  // would create output files by their "kind" rather than by path.
  SmallString<128> PathStorage(Path);
  if (FM.FixupRelativePath(PathStorage))
    return std::string(PathStorage);

  // Apply "normal" working directory.
  if (!WorkingDir.empty()) {
    SmallString<128> Tmp(Path);
    llvm::sys::fs::make_absolute(WorkingDir, Tmp);
    return std::string(Tmp);
  }
  return Path;
}

CompileJobCache::CompileJobCache() = default;
CompileJobCache::~CompileJobCache() = default;

int CompileJobCache::reportCachingBackendError(DiagnosticsEngine &Diag,
                                               Error &&E) {
  Diag.Report(diag::err_caching_backend_fail) << llvm::toString(std::move(E));
  return 1;
}

std::optional<int> CompileJobCache::initialize(CompilerInstance &Clang) {
  CompilerInvocation &Invocation = Clang.getInvocation();
  DiagnosticsEngine &Diags = Clang.getDiagnostics();
  FrontendOptions &FrontendOpts = Invocation.getFrontendOpts();
  CacheCompileJob = FrontendOpts.CacheCompileJob;

  // Nothing else to do if we're not caching.
  if (!CacheCompileJob)
    return std::nullopt;

  std::tie(CAS, Cache) = Clang.createCASDatabases();
  if (!CAS || !Cache)
    return 1; // Exit with error!

  CompilerInvocation DummyInvocation = Invocation;
  CompileJobCachingOptions DummyCacheOpts;

  CompileJobCachingOptions CacheOpts;
  auto Keys = canonicalizeAndCreateCacheKeys(*CAS, *Cache, Diags, Invocation,
                                             CacheOpts);
  if (!Keys)
    return 1; // Exit with error!
  std::tie(ResultCacheKey, ResultCacheKeyWithInputCacheKeysResolved) = *Keys;

  switch (FrontendOpts.ProgramAction) {
  case frontend::GenerateModule:
  case frontend::GenerateModuleInterface:
  case frontend::GeneratePCH:
    Clang.getPreprocessorOpts().CachingDiagOption = CachingDiagKind::Error;
    break;
  default:
    Clang.getPreprocessorOpts().CachingDiagOption = CachingDiagKind::Warning;
    break;
  }

  DisableCachedCompileJobReplay = CacheOpts.DisableCachedCompileJobReplay;
  bool UseCASBackend = Invocation.getCodeGenOpts().UseCASBackend;
  Invocation.getCodeGenOpts().MCCallBack = [&](const llvm::cas::CASID &ID) {
    MCOutputID = ID;
    return Error::success();
  };

  llvm::PrefixMapper PrefixMapper;
  llvm::SmallVector<llvm::MappedPrefix> Split;
  llvm::MappedPrefix::transformPairs(CacheOpts.PathPrefixMappings, Split);
  for (const auto &MappedPrefix : Split) {
    // We use the inverse mapping because the \p PrefixMapper will be used for
    // de-canonicalization of paths.
    PrefixMapper.add(MappedPrefix.getInverse());
  }

  if (!CacheOpts.CompilationCachingServicePath.empty()) {
    Expected<llvm::cas::remote::ClientServices> Clients =
        llvm::cas::remote::createCompilationCachingRemoteClient(
            CacheOpts.CompilationCachingServicePath);
    if (!Clients)
      return reportCachingBackendError(Clang.getDiagnostics(),
                                       Clients.takeError());
    assert(!CacheOpts.WriteOutputAsCASID &&
           "combination of options not rejected earlier?");
    assert(!UseCASBackend && "combination of options not rejected earlier?");
    CacheBackend = std::make_unique<RemoteCachingOutputs>(
        Clang, /*WorkingDir=*/"", std::move(PrefixMapper), std::move(*Clients));
  } else {
    CacheBackend = std::make_unique<ObjectStoreCachingOutputs>(
        Clang, /*WorkingDir=*/"", std::move(PrefixMapper),
        CacheOpts.WriteOutputAsCASID, UseCASBackend, MCOutputID, CAS, Cache);
  }

  return std::nullopt;
}

CompileJobCache::CachingOutputs::CachingOutputs(CompilerInstance &Clang,
                                                StringRef WorkingDir,
                                                llvm::PrefixMapper Mapper,
                                                bool WriteOutputAsCASID,
                                                bool UseCASBackend)
    : Clang(Clang), PrefixMapper(std::move(Mapper)),
      WriteOutputAsCASID(WriteOutputAsCASID), UseCASBackend(UseCASBackend) {
  CompilerInvocation &Invocation = Clang.getInvocation();
  FrontendOptions &FrontendOpts = Invocation.getFrontendOpts();
  if (!Clang.hasFileManager())
    Clang.createFileManager();
  FileManager &FM = Clang.getFileManager();
  OutputFile = fixupRelativePath(FrontendOpts.OutputFile, FM, WorkingDir);
  DependenciesFile = fixupRelativePath(
      Invocation.getDependencyOutputOpts().OutputFile, FM, WorkingDir);
  DiagProcessor = std::make_unique<clang::cas::CachingDiagnosticsProcessor>(
      PrefixMapper, FM);
}

Expected<bool> ObjectStoreCachingOutputs::tryReplayCachedResult(
    const llvm::cas::CASID &ResultCacheKey,
    const llvm::cas::CASID &CanonicalResultCacheKey) {
  DiagnosticsEngine &Diags = Clang.getDiagnostics();

  std::optional<llvm::cas::CASID> Result;
  {
    llvm::ScopedDurationTimer ScopedTime([&Diags](double Seconds) {
      Diags.Report(diag::remark_compile_job_cache_timing_backend_key_query)
          << llvm::format("%.6fs", Seconds);
    });
    // Asking for the canonical cache key gives us better chances for cache hit.
    if (Error E = Cache->get(CanonicalResultCacheKey, /*Globally=*/true)
                      .moveInto(Result))
      return std::move(E);
  }

  if (!Result) {
    Diags.Report(diag::remark_compile_job_cache_miss)
        << ResultCacheKey.toString();
    return false;
  }

  llvm::ScopedDurationTimer ScopedTime([&Diags](double Seconds) {
    Diags.Report(diag::remark_compile_job_cache_timing_backend_load)
        << llvm::format("%.6fs", Seconds);
  });

  std::optional<llvm::cas::ObjectRef> ResultRef = CAS->getReference(*Result);
  if (!ResultRef) {
    Diags.Report(diag::remark_compile_job_cache_miss_result_not_found)
        << ResultCacheKey.toString() << "result not in CAS";
    return false;
  }

  // Tasks depending on this one don't know the canonical cache key. Make sure
  // the requested key is associated with the result too.
  if (ResultCacheKey != CanonicalResultCacheKey)
    if (Error E =
            Cache->put(ResultCacheKey, *Result, /*Globally=*/true))
      return std::move(E);

  // \c replayCachedResult emits remarks for a cache hit or miss.
  std::optional<int> Status = replayCachedResult(ResultCacheKey, *ResultRef,
                                                 /*JustComputedResult=*/false);
  if (!Status)
    return false; // cache miss.
  assert(*Status == 0 && "Expected success status for a cache hit");
  return true;
}

std::optional<int>
CompileJobCache::tryReplayCachedResult(CompilerInstance &Clang) {
  if (!CacheCompileJob)
    return std::nullopt;

  DiagnosticsEngine &Diags = Clang.getDiagnostics();

  assert(ResultCacheKey.has_value() && "ResultCacheKey not initialized?");

  Clang.setCompileJobCacheKey(*ResultCacheKey);

  Expected<bool> ReplayedResult =
      DisableCachedCompileJobReplay
          ? false
          : CacheBackend->tryReplayCachedResult(
                *ResultCacheKey, *ResultCacheKeyWithInputCacheKeysResolved);
  if (!ReplayedResult)
    return reportCachingBackendError(Clang.getDiagnostics(),
                                     ReplayedResult.takeError());

  if (DisableCachedCompileJobReplay)
    Diags.Report(diag::remark_compile_job_cache_skipped)
        << ResultCacheKey->toString();

  if (*ReplayedResult)
    return 0;

  if (CacheBackend->prepareOutputCollection())
    return 1;

  return std::nullopt;
}

bool CompileJobCache::CachingOutputs::prepareOutputCollectionCommon(
    IntrusiveRefCntPtr<llvm::vfs::OutputBackend> CacheOutputs) {
  // Create an on-disk backend for streaming the results live if we run the
  // computation. If we're writing the output as a CASID, skip it here, since
  // it'll be handled during replay.
  IntrusiveRefCntPtr<llvm::vfs::OutputBackend> OnDiskOutputs =
      llvm::makeIntrusiveRefCnt<llvm::vfs::OnDiskOutputBackend>();
  if (WriteOutputAsCASID) {
    OnDiskOutputs = llvm::vfs::makeFilteringOutputBackend(
        OnDiskOutputs,
        [this](StringRef ResolvedPath, std::optional<llvm::vfs::OutputConfig>) {
          return ResolvedPath != this->OutputFile;
        });
  }

  // Set up the output backend so we can save / cache the result after.
  for (OutputKind K : clang::cas::CompileJobCacheResult::getAllOutputKinds()) {
    StringRef OutPath = getPathForOutputKind(K);
    if (!OutPath.empty())
      CachedResultBuilder.addKindMap(K, OutPath);
  }

  // When use CAS backend, filter out the output object file. Always filter out
  // the dependencies file, since we build a CAS-specific object for it.
  auto FilterBackend = llvm::vfs::makeFilteringOutputBackend(
      CacheOutputs,
      [&](StringRef Path, std::optional<llvm::vfs::OutputConfig> Config) {
        return !(UseCASBackend && (Path.str() == OutputFile)) &&
               Path != DependenciesFile;
      });

  Clang.setOutputBackend(llvm::vfs::makeMirroringOutputBackend(
      FilterBackend, std::move(OnDiskOutputs)));

  DiagProcessor->insertDiagConsumer(Clang.getDiagnostics());

  return false;
}

void CompileJobCache::CachingOutputs::stopDiagnosticsCapture() {
  DiagProcessor->removeDiagConsumer(Clang.getDiagnostics());
}

Error CompileJobCache::CachingOutputs::replayCachedDiagnostics(
    StringRef DiagsData) {
  DiagnosticConsumer &Consumer = *Clang.getDiagnostics().getClient();
  Consumer.BeginSourceFile(Clang.getLangOpts());
  if (Error E = DiagProcessor->replayCachedDiagnostics(DiagsData, Consumer))
    return E;
  Consumer.EndSourceFile();
  Clang.printDiagnosticStats();
  return Error::success();
}

bool ObjectStoreCachingOutputs::prepareOutputCollection() {
  if (prepareOutputCollectionCommon(CASOutputs))
    return true;

  if (!Clang.getDependencyOutputOpts().OutputFile.empty())
    Clang.addDependencyCollector(std::make_shared<CASDependencyCollector>(
        Clang.getDependencyOutputOpts(), *CAS,
        [this](std::optional<llvm::cas::ObjectRef> Deps) {
          DependenciesOutput = Deps;
        }));

  return false;
}

bool CompileJobCache::finishComputedResult(CompilerInstance &Clang,
                                           bool Success) {
  // Nothing to do if not caching.
  if (!CacheCompileJob)
    return Success;

  CacheBackend->stopDiagnosticsCapture();

  // Don't cache failed builds.
  //
  // TODO: Consider caching failed builds! Note: when output files are written
  // without a temporary (non-atomically), failure may cause the removal of a
  // preexisting file. That behaviour is not currently modeled by the cache.
  if (!Success)
    return false;

  DiagnosticsEngine &Diags = Clang.getDiagnostics();

  bool UnsupportedOutput;
  if (Error E = maybeIngestNonVirtualOutputFromFileSystem(Clang).moveInto(
          UnsupportedOutput)) {
    reportCachingBackendError(Diags, std::move(E));
    return false;
  }

  // Check if we encounter any source that would generate non-reproducible
  // outputs.
  bool SkipCache =
      (Clang.hasPreprocessor() && Clang.isSourceNonReproducible()) ||
      UnsupportedOutput;
  if (SkipCache) {
    switch (Clang.getPreprocessorOpts().CachingDiagOption) {
    case CachingDiagKind::None:
      break;
    case CachingDiagKind::Warning:
      Diags.Report(diag::remark_compile_job_cache_skipped)
          << ResultCacheKey->toString();
      break;
    case CachingDiagKind::Error:
      llvm_unreachable("Should not reach here if there is an error");
    }
  }

  if (Error E = CacheBackend->finishComputedResult(
          *ResultCacheKey, *ResultCacheKeyWithInputCacheKeysResolved,
          SkipCache)) {
    reportCachingBackendError(Diags, std::move(E));
    return false;
  }
  return true;
}

Expected<bool> CompileJobCache::maybeIngestNonVirtualOutputFromFileSystem(
    CompilerInstance &Clang) {
  // FIXME: All consumers should adopt \c llvm::vfs::OutputBackend and this
  // function should go away.

  const auto &FrontendOpts = Clang.getFrontendOpts();
  if (FrontendOpts.ProgramAction == frontend::RunAnalysis) {
    StringRef OutputPath = FrontendOpts.OutputFile;
    if (OutputPath.empty())
      return false;
    if (llvm::sys::fs::is_directory(OutputPath)) {
      // FIXME: A directory is produced for the 'html' output of the analyzer,
      // support it for caching purposes.
      Clang.getDiagnostics().Report(diag::warn_clang_cache_disabled_caching)
          << "analyzer output is not supported";
      return true;
    }
    if (Error E = CacheBackend->addNonVirtualOutputFile(OutputPath))
      return std::move(E);
    return false;
  }

  return false;
}

Expected<std::optional<int>> CompileJobCache::replayCachedResult(
    std::shared_ptr<CompilerInvocation> Invok, StringRef WorkingDir,
    const llvm::cas::CASID &CacheKey, cas::CompileJobCacheResult &CachedResult,
    SmallVectorImpl<char> &DiagText, bool WriteOutputAsCASID,
    std::optional<llvm::cas::CASID> *OutMCOutputID) {
  CompilerInstance Clang(std::move(Invok));
  llvm::raw_svector_ostream DiagOS(DiagText);
  Clang.createVirtualFileSystem(llvm::vfs::getRealFileSystem());
  Clang.createDiagnostics(
      new TextDiagnosticPrinter(DiagOS, Clang.getDiagnosticOpts()));
  Clang.setVerboseOutputStream(DiagOS);

  // FIXME: we should create an include-tree filesystem based on the cache key
  // to guarantee that the filesystem used during diagnostic replay will match
  // the cached diagnostics. Currently we rely on the invocation having a
  // matching -fcas-include-tree option.

  auto FinishDiagnosticClient =
      llvm::make_scope_exit([&]() { Clang.getDiagnosticClient().finish(); });

  llvm::PrefixMapper PrefixMapper;
  llvm::SmallVector<llvm::MappedPrefix> Split;
  llvm::MappedPrefix::transformPairs(
      Clang.getFrontendOpts().PathPrefixMappings, Split);
  for (const auto &MappedPrefix : Split) {
    // We use the inverse mapping because the \p PrefixMapper will be used for
    // de-canonicalization of paths.
    PrefixMapper.add(MappedPrefix.getInverse());
  }

  assert(!Clang.getDiagnostics().hasErrorOccurred());

  std::optional<llvm::cas::CASID> MCOutputID;
  ObjectStoreCachingOutputs CachingOutputs(
      Clang, WorkingDir, std::move(PrefixMapper), WriteOutputAsCASID,
      Clang.getInvocation().getCodeGenOpts().UseCASBackend, MCOutputID,
      /*CAS*/ nullptr, /*Cache*/ nullptr);
  if (OutMCOutputID)
    *OutMCOutputID = std::move(MCOutputID);

  std::optional<int> Ret;
  if (Error E = CachingOutputs
                    .replayCachedResult(CacheKey, CachedResult,
                                        /*JustComputedResult*/ false)
                    .moveInto(Ret))
    return std::move(E);

  if (Clang.getDiagnostics().hasErrorOccurred())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "error diagnostic during replay: " +
                                       DiagOS.str());

  return Ret;
}

Expected<llvm::cas::ObjectRef> ObjectStoreCachingOutputs::writeOutputs(
    const llvm::cas::CASID &ResultCacheKey) {
  DiagnosticsEngine &Diags = Clang.getDiagnostics();
  llvm::ScopedDurationTimer ScopedTime([&Diags](double Seconds) {
    Diags.Report(diag::remark_compile_job_cache_timing_backend_store)
        << llvm::format("%.6fs", Seconds);
  });

  // Add the MC output to the CAS Outputs.
  if (MCOutputID) {
    auto MCOutputRef = CAS->getReference(*MCOutputID);
    if (MCOutputRef)
      CachedResultBuilder.addOutput(OutputKind::MainOutput, *MCOutputRef);
  }

  Expected<std::optional<std::string>> SerialDiags =
      DiagProcessor->serializeEmittedDiagnostics();
  if (!SerialDiags)
    return SerialDiags.takeError();
  if (*SerialDiags) {
    Expected<llvm::cas::ObjectRef> DiagsRef =
        CAS->storeFromString({}, **SerialDiags);
    if (!DiagsRef)
      return DiagsRef.takeError();
    CachedResultBuilder.addOutput(OutputKind::SerializedDiagnostics, *DiagsRef);
  }

  if (DependenciesOutput)
    CachedResultBuilder.addOutput(OutputKind::Dependencies,
                                  *DependenciesOutput);

  auto BackendOutputs = CASOutputs->takeOutputs();
  for (auto &Output : BackendOutputs)
    if (auto Err = CachedResultBuilder.addOutput(Output.Path, Output.Object))
      return std::move(Err);

  // Cache the result.
  return CachedResultBuilder.build(*CAS);
}

Error ObjectStoreCachingOutputs::addNonVirtualOutputFile(StringRef FilePath) {
  auto F = llvm::sys::fs::openNativeFileForRead(FilePath);
  if (!F)
    return F.takeError();
  auto CloseOnExit =
      llvm::make_scope_exit([&F]() { llvm::sys::fs::closeFile(*F); });

  std::optional<llvm::cas::ObjectRef> ObjRef;
  if (Error E = CAS->storeFromOpenFile(*F).moveInto(ObjRef))
    return E;
  CASOutputs->addObject(FilePath, *ObjRef);
  return Error::success();
}

Error ObjectStoreCachingOutputs::finishComputedResult(
    const llvm::cas::CASID &ResultCacheKey,
    const llvm::cas::CASID &CanonicalResultCacheKey, bool SkipCache) {
  Expected<llvm::cas::ObjectRef> Result = writeOutputs(ResultCacheKey);
  if (!Result)
    return Result.takeError();

  // Skip caching if requested.
  if (!SkipCache) {
    DiagnosticsEngine &Diags = Clang.getDiagnostics();
    llvm::ScopedDurationTimer ScopedTime([&Diags](double Seconds) {
      Diags.Report(diag::remark_compile_job_cache_timing_backend_key_update)
          << llvm::format("%.6fs", Seconds);
    });
    if (llvm::Error E = Cache->put(CanonicalResultCacheKey, CAS->getID(*Result),
                                   /*Globally=*/true))
      return E;
    // Tasks depending on this one don't know the canonical cache key. Make sure
    // the requested key is associated with the result too.
    if (ResultCacheKey != CanonicalResultCacheKey) {
      if (llvm::Error E = Cache->put(ResultCacheKey, CAS->getID(*Result),
                                     /*Globally=*/true))
        return E;
    }
  }

  // Replay / decanonicalize as necessary.
  std::optional<int> Status = replayCachedResult(ResultCacheKey, *Result,
                                                 /*JustComputedResult=*/true);
  (void)Status;
  assert(Status == std::nullopt);
  return Error::success();
}

/// Replay a result after a cache hit.
std::optional<int> ObjectStoreCachingOutputs::replayCachedResult(
    const llvm::cas::CASID &ResultCacheKey, llvm::cas::ObjectRef ResultID,
    bool JustComputedResult) {
  if (JustComputedResult && !WriteOutputAsCASID)
    return std::nullopt;

  // FIXME: Stop calling report_fatal_error().
  std::optional<clang::cas::CompileJobCacheResult> Result;
  clang::cas::CompileJobResultSchema Schema(*CAS);
  if (Error E = Schema.load(ResultID).moveInto(Result))
    llvm::report_fatal_error(std::move(E));

  std::optional<int> Ret;
  // FIXME: Stop calling report_fatal_error().
  if (Error E = replayCachedResult(ResultCacheKey, *Result, JustComputedResult)
                    .moveInto(Ret))
    llvm::report_fatal_error(std::move(E));

  return Ret;
}

Expected<std::optional<int>> ObjectStoreCachingOutputs::replayCachedResult(
    const llvm::cas::CASID &ResultCacheKey,
    clang::cas::CompileJobCacheResult &Result, bool JustComputedResult) {
  if (JustComputedResult && !WriteOutputAsCASID)
    return std::nullopt;

  llvm::cas::ObjectStore &CAS = Result.getCAS();
  DiagnosticsEngine &Diags = Clang.getDiagnostics();
  bool HasMissingOutput = false;
  std::optional<llvm::cas::ObjectProxy> SerialDiags;
  llvm::vfs::OnDiskOutputBackend Backend;

  auto processOutput = [&](clang::cas::CompileJobCacheResult::Output O,
                           std::optional<llvm::cas::ObjectProxy> Obj) -> Error {
    if (!Obj.has_value()) {
      Diags.Report(diag::remark_compile_job_cache_backend_output_not_found)
          << clang::cas::CompileJobCacheResult::getOutputKindName(O.Kind)
          << ResultCacheKey.toString() << CAS.getID(O.Object).toString();
      HasMissingOutput = true;
      return Error::success();
    }
    if (HasMissingOutput)
      return Error::success();

    if (O.Kind == OutputKind::SerializedDiagnostics) {
      SerialDiags = Obj;
      return Error::success();
    }

    std::string Path = std::string(getPathForOutputKind(O.Kind));
    if (Path.empty())
      // The output may be always generated but not needed with this invocation.
      return Error::success(); // continue

    // Always create parent directory of outputs, since it is hard to precisely
    // match which outputs rely on creating parents and the order outputs are
    // replayed in, in case a previous output would create the parent
    // (e.g. a .pcm and .diag file in the same directory).
    StringRef ParentPath = llvm::sys::path::parent_path(Path);
    if (!ParentPath.empty())
      llvm::sys::fs::create_directories(ParentPath);

    bool IsOutputFile = O.Kind == OutputKind::MainOutput;

    if (IsOutputFile && ComputedJobNeedsReplay) {
      auto Output = Backend.createFile(Path);
      if (!Output)
        return Output.takeError();
      if (WriteOutputAsCASID)
        llvm::cas::writeCASIDBuffer(CAS.getID(O.Object), *Output);
      else if (UseCASBackend) {
        // Replay by write out object file.
        // When the environmental variable is set, save the backend CASID for
        // analysis later.
        if (llvm::sys::Process::GetEnv("CLANG_CAS_BACKEND_SAVE_CASID_FILE")) {
          std::string CASIDPath = Path + ".casid";
          std::error_code EC;
          auto IDOut = Backend.createFile(CASIDPath);
          if (!IDOut)
            return IDOut.takeError();
          writeCASIDBuffer(CAS.getID(O.Object), *IDOut);
          if (auto E = IDOut->keep())
            return E;
        }
        auto Schema = std::make_unique<llvm::mccasformats::v1::MCSchema>(CAS);
        if (auto E = Schema->serializeObjectFile(*Obj, *Output))
          return E;
      }
      return Output->keep();
    }

    if (JustComputedResult)
      return Error::success(); // continue

    auto Output = Backend.createFile(Path);
    if (!Output)
      return Output.takeError();
    if (O.Kind == OutputKind::Dependencies) {
      if (auto E = CASDependencyCollector::replay(
              Clang.getDependencyOutputOpts(), CAS, *Obj, *Output))
        return E;
    } else {
      *Output << Obj->getData();
    }

    return Output->keep();
  };

  if (auto Err = Result.forEachLoadedOutput(processOutput))
    return std::move(Err);

  if (HasMissingOutput) {
    Diags.Report(diag::remark_compile_job_cache_miss)
        << ResultCacheKey.toString();
    return std::nullopt;
  }

  if (!JustComputedResult) {
    Diags.Report(diag::remark_compile_job_cache_hit)
        << ResultCacheKey.toString() << Result.getID().toString();

    if (SerialDiags) {
      if (Error E = replayCachedDiagnostics(SerialDiags->getData()))
        return std::move(E);
    }
  }

  if (JustComputedResult)
    return std::nullopt;
  return 0;
}

Expected<bool> RemoteCachingOutputs::tryReplayCachedResult(
    const llvm::cas::CASID &ResultCacheKey,
    const llvm::cas::CASID &CanonicalResultCacheKey) {
  DiagnosticsEngine &Diags = Clang.getDiagnostics();

  std::optional<
      llvm::cas::remote::KeyValueDBClient::GetValueAsyncQueue::Response>
      Response;
  {
    llvm::ScopedDurationTimer ScopedTime([&Diags](double Seconds) {
      Diags.Report(diag::remark_compile_job_cache_timing_backend_key_query)
          << llvm::format("%.6fs", Seconds);
    });
    RemoteKVClient->getValueQueue().getValueAsync(
        CanonicalResultCacheKey.getHash());
    if (Error E =
            RemoteKVClient->getValueQueue().receiveNext().moveInto(Response))
      return std::move(E);
  }
  if (!Response->Value) {
    Diags.Report(diag::remark_compile_job_cache_miss)
        << ResultCacheKey.toString();
    return false;
  }

  // Tasks depending on this one don't know the canonical cache key. Make sure
  // the requested key is associated with the result too.
  if (ResultCacheKey != CanonicalResultCacheKey) {
    RemoteKVClient->putValueQueue().putValueAsync(ResultCacheKey.getHash(),
                                                  *Response->Value);
    auto PutResponse = RemoteKVClient->putValueQueue().receiveNext();
    if (!PutResponse)
      return PutResponse.takeError();
  }

  llvm::ScopedDurationTimer ScopedTime([&Diags](double Seconds) {
    Diags.Report(diag::remark_compile_job_cache_timing_backend_load)
        << llvm::format("%.6fs", Seconds);
  });

  Expected<bool> ReplayedResult =
      replayCachedResult(ResultCacheKey, *Response->Value);
  if (!ReplayedResult)
    return ReplayedResult.takeError();

  // diag::remark_compile_job_cache_hit is emitted in \p replayCachedResult.

  return ReplayedResult;
}

static constexpr llvm::StringLiteral MainOutputKindName = "<output>";
static constexpr llvm::StringLiteral SerializedDiagnosticsKindName =
    "<serial-diags>";
static constexpr llvm::StringLiteral DependenciesOutputKindName =
    "<dependencies>";

StringRef RemoteCachingOutputs::getOutputKindName(OutputKind Kind) {
  switch (Kind) {
  case OutputKind::MainOutput:
    return MainOutputKindName;
  case OutputKind::SerializedDiagnostics:
    return SerializedDiagnosticsKindName;
  case OutputKind::Dependencies:
    return DependenciesOutputKindName;
  }
}

std::optional<CompileJobCache::OutputKind>
RemoteCachingOutputs::getOutputKindForName(StringRef Name) {
  return llvm::StringSwitch<std::optional<OutputKind>>(Name)
      .Case(MainOutputKindName, OutputKind::MainOutput)
      .Case(SerializedDiagnosticsKindName, OutputKind::SerializedDiagnostics)
      .Case(DependenciesOutputKindName, OutputKind::Dependencies)
      .Default(std::nullopt);
}

Expected<bool> RemoteCachingOutputs::replayCachedResult(
    const llvm::cas::CASID &ResultCacheKey,
    const llvm::cas::remote::KeyValueDBClient::ValueTy &CompResult) {
  // It would be nice to release the llbuild execution lane while we wait to
  // receive remote data, but if some data are missing (e.g. due to garbage
  // collection), we'll fallback to normal compilation and it would be badness
  // to do it outside the execution lanes.
  // FIXME: Consider enhancing the llbuild interaction to allow "requesting
  // back" an execution lane, then we would release the execution lane here and
  // if we need to fallback to normal compilation we'd ask and wait for an
  // execution lane before continuing it.

  // Replay outputs.

  DiagnosticsEngine &Diags = Clang.getDiagnostics();

  auto &LoadQueue = RemoteCASClient->loadQueue();
  struct CallCtx : public llvm::cas::remote::AsyncCallerContext {
    StringRef OutputName;
    StringRef CASID;
    bool IsDiags;
    CallCtx(StringRef OutputName, StringRef CASID, bool IsDiags)
        : OutputName(OutputName), CASID(CASID), IsDiags(IsDiags) {}
  };
  auto makeCtx =
      [](StringRef OutputName, StringRef CASID,
         bool IsStderr =
             false) -> std::shared_ptr<llvm::cas::remote::AsyncCallerContext> {
    return std::make_shared<CallCtx>(OutputName, CASID, IsStderr);
  };

  for (const auto &Entry : CompResult) {
    StringRef OutputName = Entry.first();
    const std::string &CASID = Entry.second;

    std::optional<OutputKind> OutKind = getOutputKindForName(OutputName);
    StringRef Path = OutKind ? getPathForOutputKind(*OutKind) : OutputName;

    if (OutKind && *OutKind == OutputKind::SerializedDiagnostics) {
      LoadQueue.loadAsync(CASID, /*OutFilePath*/ std::nullopt,
                          makeCtx(OutputName, CASID, /*IsDiags*/ true));
      continue;
    }
    if (Path.empty()) {
      // The output may be always generated but not needed with this invocation,
      // like the serialized diagnostics file.
      continue;
    }
    LoadQueue.loadAsync(CASID, Path.str(), makeCtx(OutputName, CASID));
  }

  bool HasMissingOutput = false;
  std::optional<std::string> SerialDiags;

  while (LoadQueue.hasPending()) {
    auto Response = LoadQueue.receiveNext();
    if (!Response)
      return Response.takeError();
    const CallCtx &Ctx = *static_cast<CallCtx *>(Response->CallCtx.get());
    if (Response->KeyNotFound) {
      std::string PrintedRemoteCASID = getPrintableRemoteID(Ctx.CASID);
      Diags.Report(diag::remark_compile_job_cache_backend_output_not_found)
          << Ctx.OutputName << ResultCacheKey.toString() << PrintedRemoteCASID;
      HasMissingOutput = true;
      continue;
    }
    if (HasMissingOutput)
      continue;

    if (Ctx.IsDiags)
      SerialDiags = std::move(Response->BlobData);
  }

  if (HasMissingOutput)
    return false;

  StringRef MainOutputName = getOutputKindName(OutputKind::MainOutput);
  auto MainOutputI = CompResult.find(MainOutputName);
  assert(MainOutputI != CompResult.end());
  std::string PrintedRemoteMainOutputCASID =
      getPrintableRemoteID(MainOutputI->second);
  Diags.Report(diag::remark_compile_job_cache_hit)
      << ResultCacheKey.toString()
      << (Twine(MainOutputName) + ": " + PrintedRemoteMainOutputCASID).str();

  if (SerialDiags) {
    if (Error E = replayCachedDiagnostics(*SerialDiags))
      return std::move(E);
  }

  return true;
}

bool RemoteCachingOutputs::prepareOutputCollection() {
  // Set up the output backend so we can save / cache the result after.
  for (OutputKind K : clang::cas::CompileJobCacheResult::getAllOutputKinds()) {
    StringRef OutPath = getPathForOutputKind(K);
    if (!OutPath.empty())
      CollectingOutputs->addKindMap(getOutputKindName(K), OutPath);
  }

  // FIXME: Handle collecting the dependencies as well.
  return prepareOutputCollectionCommon(CollectingOutputs);
}

Expected<llvm::cas::remote::KeyValueDBClient::ValueTy>
RemoteCachingOutputs::saveOutputs(const llvm::cas::CASID &ResultCacheKey) {
  DiagnosticsEngine &Diags = Clang.getDiagnostics();
  llvm::ScopedDurationTimer ScopedTime([&Diags](double Seconds) {
    Diags.Report(diag::remark_compile_job_cache_timing_backend_store)
        << llvm::format("%.6fs", Seconds);
  });

  auto &SaveQueue = RemoteCASClient->saveQueue();
  struct CallCtx : public llvm::cas::remote::AsyncCallerContext {
    StringRef OutputName;
    CallCtx(StringRef OutputName) : OutputName(OutputName) {}
  };
  auto makeCtx = [](StringRef OutputName)
      -> std::shared_ptr<llvm::cas::remote::AsyncCallerContext> {
    return std::make_shared<CallCtx>(OutputName);
  };

  Expected<std::optional<std::string>> SerialDiags =
      DiagProcessor->serializeEmittedDiagnostics();
  if (!SerialDiags)
    return SerialDiags.takeError();
  if (*SerialDiags) {
    SaveQueue.saveDataAsync(
        std::move(**SerialDiags),
        makeCtx(getOutputKindName(OutputKind::SerializedDiagnostics)));
  }

  // FIXME: Save dependencies output.

  for (StringRef OutputName : CollectingOutputs->getOutputs()) {
    std::optional<OutputKind> OutKind = getOutputKindForName(OutputName);
    StringRef Path = OutKind ? getPathForOutputKind(*OutKind) : OutputName;
    assert(!Path.empty());
    SmallString<256> AbsPath{Path};
    llvm::sys::fs::make_absolute(AbsPath);
    SaveQueue.saveFileAsync(AbsPath.str().str(), makeCtx(OutputName));
  }

  // Cache the result.

  llvm::cas::remote::KeyValueDBClient::ValueTy CompResult;
  while (SaveQueue.hasPending()) {
    auto Response = SaveQueue.receiveNext();
    if (!Response)
      return Response.takeError();
    StringRef OutputName =
        static_cast<CallCtx *>(Response->CallCtx.get())->OutputName;
    CompResult[OutputName] = Response->CASID;
  }

  return std::move(CompResult);
}

Error RemoteCachingOutputs::addNonVirtualOutputFile(StringRef FilePath) {
  CollectingOutputs->addOutput(FilePath);
  return Error::success();
}

Error RemoteCachingOutputs::finishComputedResult(
    const llvm::cas::CASID &ResultCacheKey,
    const llvm::cas::CASID &CanonicalResultCacheKey, bool SkipCache) {
  if (SkipCache)
    return Error::success();

  // Release the llbuild execution lane while we wait to upload data to remote
  // cache.
  tryReleaseLLBuildExecutionLane();

  Expected<llvm::cas::remote::KeyValueDBClient::ValueTy> CompResult =
      saveOutputs(ResultCacheKey);
  if (!CompResult)
    return CompResult.takeError();

  DiagnosticsEngine &Diags = Clang.getDiagnostics();
  llvm::ScopedDurationTimer ScopedTime([&Diags](double Seconds) {
    Diags.Report(diag::remark_compile_job_cache_timing_backend_key_update)
        << llvm::format("%.6fs", Seconds);
  });

  RemoteKVClient->putValueQueue().putValueAsync(
      CanonicalResultCacheKey.getHash(), *CompResult);
  // Tasks depending on this one don't know the canonical cache key. Make sure
  // the requested key is associated with the result too.
  if (ResultCacheKey != CanonicalResultCacheKey)
    RemoteKVClient->putValueQueue().putValueAsync(ResultCacheKey.getHash(),
                                                  *CompResult);

  auto Response = RemoteKVClient->putValueQueue().receiveNext();
  if (!Response)
    return Response.takeError();
  // Tasks depending on this one don't know the canonical cache key. Make sure
  // the requested key is associated with the result too.
  if (ResultCacheKey != CanonicalResultCacheKey) {
    auto Response2 = RemoteKVClient->putValueQueue().receiveNext();
    if (!Response2)
      return Response2.takeError();
  }

  return Error::success();
}

std::string
RemoteCachingOutputs::getPrintableRemoteID(StringRef RemoteCASIDBytes) {
  // FIXME: Enhance the remote protocol for the service to be able to provide
  // a string suitable for logging a remote CASID.
  return "<remote-ID>";
}

void RemoteCachingOutputs::tryReleaseLLBuildExecutionLane() {
  if (TriedReleaseLLBuildExecutionLane)
    return;
  TriedReleaseLLBuildExecutionLane = true;
  if (auto LLTaskID = llvm::sys::Process::GetEnv("LLBUILD_TASK_ID")) {
    // Use the llbuild protocol to request to release the execution lane for
    // this task.
    auto LLControlFD = llvm::sys::Process::GetEnv("LLBUILD_CONTROL_FD");
    if (!LLControlFD)
      return; // LLBUILD_CONTROL_FD may not be set if a shell script is invoked.
    int LLCtrlFD;
    bool HasErr = StringRef(*LLControlFD).getAsInteger(10, LLCtrlFD);
    if (HasErr)
      llvm::report_fatal_error(Twine("failed converting 'LLBUILD_CONTROL_FD' "
                                     "to an integer, it was: ") +
                               *LLControlFD);
    llvm::raw_fd_ostream FDOS(LLCtrlFD, /*shouldClose*/ false);
    FDOS << "llbuild.1\n" << LLTaskID << '\n';
    FDOS.flush();
  }
}
