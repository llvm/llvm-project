//===-- cc1_main.cpp - Clang CC1 Compiler Frontend ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the entry point to the clang -cc1 functionality, which implements the
// core compiler functionality along with a number of additional tools for
// demonstration and testing purposes.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/DiagnosticCAS.h"
#include "clang/Basic/Stack.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/Basic/Version.h"
#include "clang/CodeGen/ObjectFilePCHContainerOperations.h"
#include "clang/Config/config.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/Options.h"
#include "clang/Frontend/CASDependencyCollector.h"
#include "clang/Frontend/ChainedDiagnosticConsumer.h"
#include "clang/Frontend/CompileJobCacheKey.h"
#include "clang/Frontend/CompileJobCacheResult.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Frontend/SerializedDiagnosticPrinter.h"
#include "clang/Frontend/TextDiagnosticBuffer.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Frontend/Utils.h"
#include "clang/FrontendTool/Utils.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CAS/ActionCache.h"
#include "llvm/CAS/CASFileSystem.h"
#include "llvm/CAS/CASOutputBackend.h"
#include "llvm/CAS/CASReference.h"
#include "llvm/CAS/HierarchicalTreeBuilder.h"
#include "llvm/CAS/ObjectStore.h"
#include "llvm/CAS/TreeSchema.h"
#include "llvm/CAS/Utils.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/LinkAllPasses.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/RemoteCachingService/Client.h"
#include "llvm/Support/BuryPointer.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileOutputBuffer.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/ScopedDurationTimer.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/TimeProfiler.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/VirtualOutputBackends.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include <cstdio>

#ifdef CLANG_HAVE_RLIMITS
#include <sys/resource.h>
#endif

using namespace clang;
using namespace llvm::opt;
using llvm::Error;

//===----------------------------------------------------------------------===//
// Main driver
//===----------------------------------------------------------------------===//

static void LLVMErrorHandler(void *UserData, const char *Message,
                             bool GenCrashDiag) {
  DiagnosticsEngine &Diags = *static_cast<DiagnosticsEngine*>(UserData);

  Diags.Report(diag::err_fe_error_backend) << Message;

  // Run the interrupt handlers to make sure any special cleanups get done, in
  // particular that we remove files registered with RemoveFileOnSignal.
  llvm::sys::RunInterruptHandlers();

  // We cannot recover from llvm errors.  When reporting a fatal error, exit
  // with status 70 to generate crash diagnostics.  For BSD systems this is
  // defined as an internal software error.  Otherwise, exit with status 1.
  llvm::sys::Process::Exit(GenCrashDiag ? 70 : 1);
}

#ifdef CLANG_HAVE_RLIMITS
#if defined(__linux__) && defined(__PIE__)
static size_t getCurrentStackAllocation() {
  // If we can't compute the current stack usage, allow for 512K of command
  // line arguments and environment.
  size_t Usage = 512 * 1024;
  if (FILE *StatFile = fopen("/proc/self/stat", "r")) {
    // We assume that the stack extends from its current address to the end of
    // the environment space. In reality, there is another string literal (the
    // program name) after the environment, but this is close enough (we only
    // need to be within 100K or so).
    unsigned long StackPtr, EnvEnd;
    // Disable silly GCC -Wformat warning that complains about length
    // modifiers on ignored format specifiers. We want to retain these
    // for documentation purposes even though they have no effect.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat"
#endif
    if (fscanf(StatFile,
               "%*d %*s %*c %*d %*d %*d %*d %*d %*u %*lu %*lu %*lu %*lu %*lu "
               "%*lu %*ld %*ld %*ld %*ld %*ld %*ld %*llu %*lu %*ld %*lu %*lu "
               "%*lu %*lu %lu %*lu %*lu %*lu %*lu %*lu %*llu %*lu %*lu %*d %*d "
               "%*u %*u %*llu %*lu %*ld %*lu %*lu %*lu %*lu %*lu %*lu %lu %*d",
               &StackPtr, &EnvEnd) == 2) {
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
      Usage = StackPtr < EnvEnd ? EnvEnd - StackPtr : StackPtr - EnvEnd;
    }
    fclose(StatFile);
  }
  return Usage;
}

#include <alloca.h>

LLVM_ATTRIBUTE_NOINLINE
static void ensureStackAddressSpace() {
  // Linux kernels prior to 4.1 will sometimes locate the heap of a PIE binary
  // relatively close to the stack (they are only guaranteed to be 128MiB
  // apart). This results in crashes if we happen to heap-allocate more than
  // 128MiB before we reach our stack high-water mark.
  //
  // To avoid these crashes, ensure that we have sufficient virtual memory
  // pages allocated before we start running.
  size_t Curr = getCurrentStackAllocation();
  const int kTargetStack = DesiredStackSize - 256 * 1024;
  if (Curr < kTargetStack) {
    volatile char *volatile Alloc =
        static_cast<volatile char *>(alloca(kTargetStack - Curr));
    Alloc[0] = 0;
    Alloc[kTargetStack - Curr - 1] = 0;
  }
}
#else
static void ensureStackAddressSpace() {}
#endif

/// Attempt to ensure that we have at least 8MiB of usable stack space.
static void ensureSufficientStack() {
  struct rlimit rlim;
  if (getrlimit(RLIMIT_STACK, &rlim) != 0)
    return;

  // Increase the soft stack limit to our desired level, if necessary and
  // possible.
  if (rlim.rlim_cur != RLIM_INFINITY &&
      rlim.rlim_cur < rlim_t(DesiredStackSize)) {
    // Try to allocate sufficient stack.
    if (rlim.rlim_max == RLIM_INFINITY ||
        rlim.rlim_max >= rlim_t(DesiredStackSize))
      rlim.rlim_cur = DesiredStackSize;
    else if (rlim.rlim_cur == rlim.rlim_max)
      return;
    else
      rlim.rlim_cur = rlim.rlim_max;

    if (setrlimit(RLIMIT_STACK, &rlim) != 0 ||
        rlim.rlim_cur != DesiredStackSize)
      return;
  }

  // We should now have a stack of size at least DesiredStackSize. Ensure
  // that we can actually use that much, if necessary.
  ensureStackAddressSpace();
}
#else
static void ensureSufficientStack() {}
#endif

/// Print supported cpus of the given target.
static int PrintSupportedCPUs(std::string TargetStr) {
  std::string Error;
  const llvm::Target *TheTarget =
      llvm::TargetRegistry::lookupTarget(TargetStr, Error);
  if (!TheTarget) {
    llvm::errs() << Error;
    return 1;
  }

  // the target machine will handle the mcpu printing
  llvm::TargetOptions Options;
  std::unique_ptr<llvm::TargetMachine> TheTargetMachine(
      TheTarget->createTargetMachine(TargetStr, "", "+cpuhelp", Options, None));
  return 0;
}

namespace {

/// Represents a mechanism for storing and retrieving compilation artifacts.
/// It includes common functionality and extension points for specific backend
/// implementations.
class CachingOutputs {
public:
  using OutputKind = cas::CompileJobCacheResult::OutputKind;

  CachingOutputs(CompilerInstance &Clang);
  virtual ~CachingOutputs() = default;

  /// \returns true if result was found and replayed, false otherwise.
  virtual Expected<bool>
  tryReplayCachedResult(const llvm::cas::CASID &ResultCacheKey) = 0;

  /// \returns true on failure, false on success.
  virtual bool prepareOutputCollection() = 0;

  /// Finish writing outputs from a computed result, after a cache miss.
  /// If SkipCache is true, it should not insert the ResultCacheKey into
  /// Cache for future uses.
  virtual Error finishComputedResult(const llvm::cas::CASID &ResultCacheKey,
                                     bool SkipCache) = 0;

  void finishSerializedDiagnostics();

protected:
  StringRef getPathForOutputKind(OutputKind Kind);

  bool prepareOutputCollectionCommon(
      IntrusiveRefCntPtr<llvm::vfs::OutputBackend> CacheOutputs);

  CompilerInstance &Clang;
  cas::CompileJobCacheResult::Builder CachedResultBuilder;
  std::string OutputFile;
  std::string SerialDiagsFile;
  std::string DependenciesFile;
  SmallString<256> ResultDiags;
  std::unique_ptr<llvm::raw_ostream> ResultDiagsOS;
  SmallString<256> SerialDiagsBuf;
  Optional<llvm::vfs::OutputFile> SerialDiagsOutput;
};

/// Store and retrieve compilation artifacts using \p llvm::cas::ObjectStore and
/// \p llvm::cas::ActionCache.
class ObjectStoreCachingOutputs : public CachingOutputs {
public:
  ObjectStoreCachingOutputs(CompilerInstance &Clang,
                            std::shared_ptr<llvm::cas::ObjectStore> DB,
                            std::shared_ptr<llvm::cas::ActionCache> Cache)
      : CachingOutputs(Clang), CAS(std::move(DB)), Cache(std::move(Cache)) {
    CASOutputs = llvm::makeIntrusiveRefCnt<llvm::cas::CASOutputBackend>(*CAS);
  }

private:
  Expected<bool>
  tryReplayCachedResult(const llvm::cas::CASID &ResultCacheKey) override;

  bool prepareOutputCollection() override;

  Error finishComputedResult(const llvm::cas::CASID &ResultCacheKey,
                             bool SkipCache) override;

  Expected<cas::ObjectRef> writeOutputs(const llvm::cas::CASID &ResultCacheKey);

  /// Replay a cache hit.
  ///
  /// Return status if should exit immediately, otherwise None.
  Optional<int> replayCachedResult(llvm::cas::ObjectRef ResultID,
                                   bool JustComputedResult);

  std::shared_ptr<llvm::cas::ObjectStore> CAS;
  std::shared_ptr<llvm::cas::ActionCache> Cache;
  IntrusiveRefCntPtr<llvm::cas::CASOutputBackend> CASOutputs;
  Optional<llvm::cas::ObjectRef> DependenciesOutput;
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
                 Optional<llvm::vfs::OutputConfig> Config) override {
    StringRef Name = tryRemapPath(Path);
    OutputNames.push_back(Name.str());
    return ProxyOutputBackend::createFileImpl(Path, std::move(Config));
  }

  IntrusiveRefCntPtr<llvm::vfs::OutputBackend> cloneImpl() const override {
    return IntrusiveRefCntPtr<CollectingOutputBackend>(
        const_cast<CollectingOutputBackend *>(this));
  }
};

/// Store and retrieve compilation artifacts using \p llvm::cas::CASDBClient
/// and \p llvm::cas::KeyValueDBClient.
class RemoteCachingOutputs : public CachingOutputs {
public:
  RemoteCachingOutputs(CompilerInstance &Clang,
                       llvm::cas::remote::ClientServices Clients)
      : CachingOutputs(Clang) {
    RemoteKVClient = std::move(Clients.KVDB);
    RemoteCASClient = std::move(Clients.CASDB);
    CollectingOutputs = llvm::makeIntrusiveRefCnt<CollectingOutputBackend>();
  }

private:
  Expected<bool>
  tryReplayCachedResult(const llvm::cas::CASID &ResultCacheKey) override;

  bool prepareOutputCollection() override;

  Error finishComputedResult(const llvm::cas::CASID &ResultCacheKey,
                             bool SkipCache) override;

  Expected<llvm::cas::remote::KeyValueDBClient::ValueTy>
  saveOutputs(const llvm::cas::CASID &ResultCacheKey);

  Expected<bool> replayCachedResult(
      const llvm::cas::CASID &ResultCacheKey,
      const llvm::cas::remote::KeyValueDBClient::ValueTy &CompResult);

  std::string getPrintableRemoteID(StringRef RemoteCASIDBytes);

  void tryReleaseLLBuildExecutionLane();

  static StringRef getOutputKindName(OutputKind Kind);
  /// \returns \p None if \p Name doesn't match one of the output kind names.
  static Optional<OutputKind> getOutputKindForName(StringRef Name);

  std::unique_ptr<llvm::cas::remote::KeyValueDBClient> RemoteKVClient;
  std::unique_ptr<llvm::cas::remote::CASDBClient> RemoteCASClient;
  IntrusiveRefCntPtr<CollectingOutputBackend> CollectingOutputs;
  bool TriedReleaseLLBuildExecutionLane = false;
};

// Manage caching and replay of compile jobs.
//
// The high-level model is:
//
//  1. Extract options from the CompilerInvocation:
//       - that can be simulated and
//       - that don't affect the compile job's result.
//  2. Canonicalize the options extracted in (1).
//  3. Compute the result of the compile job using the canonicalized
//     CompilerInvocation, with hooks installed to redirect outputs and
//     enable live-streaming of a running compile job to stdout or stderr.
//       - Compute a cache key.
//       - Check the cache, and run the compile job if there's a cache miss.
//       - Store the result of the compile job in the cache.
//  4. Replay the compile job, using the options extracted in (1).
//
// An example (albeit not yet implemented) is handling options controlling
// output of diagnostics. The CompilerInvocation can be canonicalized to
// serialize the diagnostics to a virtual path (<output>.diag or something).
//
//   - On a cache miss, the compile job runs, and the diagnostics are
//     serialized and stored in the cache per the canonicalized options
//     from (2).
//   - Either way, the diagnostics are replayed according to the options
//     extracted from (1) during (4).
//
// The above will produce the correct output for diagnostics, but the experience
// will be degraded in the common command-line case (emitting to stderr)
// because the diagnostics will not be streamed live. This can be improved:
//
//   - Change (3) to accept a hook: a DiagnosticsConsumer that diagnostics
//     are mirrored to (in addition to canonicalized options from (2)).
//   - If diagnostics would be live-streamed, send in a diagnostics consumer
//     that matches (1). Otherwise, send in an IgnoringDiagnosticsConsumer.
//   - In step (4), only skip replaying the diagnostics if they were already
//     handled.
class CompileJobCache {
public:
  using OutputKind = cas::CompileJobCacheResult::OutputKind;

  StringRef getPathForOutputKind(OutputKind Kind);

  /// Canonicalize \p Clang.
  ///
  /// \returns status if should exit immediately, otherwise None.
  ///
  /// TODO: Refactor \a cc1_main() so that instead this canonicalizes the
  /// CompilerInvocation before Clang gets access to command-line arguments, to
  /// control what might leak.
  Optional<int> initialize(CompilerInstance &Clang);

  /// Try looking up a cached result and replaying it.
  ///
  /// \returns status if should exit immediately, otherwise None.
  Optional<int> tryReplayCachedResult(CompilerInstance &Clang);

  /// Finish writing outputs from a computed result, after a cache miss.
  ///
  /// \returns true if finished successfully.
  bool finishComputedResult(CompilerInstance &Clang, bool Success);

private:
  int reportCachingBackendError(DiagnosticsEngine &Diag, Error &&E) {
    Diag.Report(diag::err_caching_backend_fail) << llvm::toString(std::move(E));
    return 1;
  }

  bool CacheCompileJob = false;
  bool DisableCachedCompileJobReplay = false;

  std::shared_ptr<llvm::cas::ObjectStore> CAS;
  std::shared_ptr<llvm::cas::ActionCache> Cache;
  Optional<llvm::cas::CASID> ResultCacheKey;

  std::unique_ptr<CachingOutputs> CacheBackend;
};
} // end anonymous namespace

StringRef CachingOutputs::getPathForOutputKind(OutputKind Kind) {
  switch (Kind) {
  case OutputKind::MainOutput:
    return OutputFile;
  case OutputKind::SerializedDiagnostics:
    return SerialDiagsFile;
  case OutputKind::Dependencies:
    return DependenciesFile;
  default:
    return "";
  }
}

static std::string fixupRelativePath(const std::string &Path, FileManager &FM) {
  // FIXME: this needs to stay in sync with createOutputFileImpl. Ideally, clang
  // would create output files by their "kind" rather than by path.
  if (!Path.empty() && Path != "-" && !llvm::sys::path::is_absolute(Path)) {
    SmallString<128> PathStorage(Path);
    if (FM.FixupRelativePath(PathStorage))
      return std::string(PathStorage);
  }
  return Path;
}

Optional<int> CompileJobCache::initialize(CompilerInstance &Clang) {
  CompilerInvocation &Invocation = Clang.getInvocation();
  DiagnosticsEngine &Diags = Clang.getDiagnostics();
  FrontendOptions &FrontendOpts = Invocation.getFrontendOpts();
  CacheCompileJob = FrontendOpts.CacheCompileJob;

  // Nothing else to do if we're not caching.
  if (!CacheCompileJob)
    return None;

  CAS = Invocation.getCASOpts().getOrCreateObjectStore(Diags);
  if (!CAS)
    return 1; // Exit with error!
  Cache = Invocation.getCASOpts().getOrCreateActionCache(Diags);
  if (!Cache)
    return 1; // Exit with error!

  CompileJobCachingOptions CacheOpts;
  ResultCacheKey =
      canonicalizeAndCreateCacheKey(*CAS, Diags, Invocation, CacheOpts);
  if (!ResultCacheKey)
    return 1; // Exit with error!

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

  if (!CacheOpts.CompilationCachingServicePath.empty()) {
    Expected<llvm::cas::remote::ClientServices> Clients =
        llvm::cas::remote::createCompilationCachingRemoteClient(
            CacheOpts.CompilationCachingServicePath);
    if (!Clients)
      return reportCachingBackendError(Clang.getDiagnostics(),
                                       Clients.takeError());
    CacheBackend =
        std::make_unique<RemoteCachingOutputs>(Clang, std::move(*Clients));
  } else {
    CacheBackend =
        std::make_unique<ObjectStoreCachingOutputs>(Clang, CAS, Cache);
  }

  return None;
}

CachingOutputs::CachingOutputs(CompilerInstance &Clang) : Clang(Clang) {
  CompilerInvocation &Invocation = Clang.getInvocation();
  FrontendOptions &FrontendOpts = Invocation.getFrontendOpts();
  if (!Clang.hasFileManager())
    Clang.createFileManager();
  FileManager &FM = Clang.getFileManager();
  OutputFile = fixupRelativePath(FrontendOpts.OutputFile, FM);
  SerialDiagsFile = fixupRelativePath(
      Invocation.getDiagnosticOpts().DiagnosticSerializationFile, FM);
  DependenciesFile =
      fixupRelativePath(Invocation.getDependencyOutputOpts().OutputFile, FM);
}

namespace {
class raw_mirroring_ostream : public llvm::raw_ostream {
  llvm::raw_ostream &Base;
  std::unique_ptr<llvm::raw_ostream> Reflection;

  void write_impl(const char *Ptr, size_t Size) override {
    Base.write(Ptr, Size);
    Reflection->write(Ptr, Size);
  }

  uint64_t current_pos() const override { return Base.tell(); }

public:
  raw_mirroring_ostream(llvm::raw_ostream &Base,
                        std::unique_ptr<llvm::raw_ostream> Reflection)
      : Base(Base), Reflection(std::move(Reflection)) {
    // FIXME: Is this right?
    enable_colors(true);
    SetUnbuffered();
  }

  bool is_displayed() const override { return Base.is_displayed(); }

  bool has_colors() const override { return Base.has_colors(); }
};
} // namespace

static Expected<llvm::vfs::OutputFile>
createBinaryOutputFile(CompilerInstance &Clang, StringRef OutputPath) {
  using namespace llvm::vfs;
  Expected<OutputFile> O = Clang.getOrCreateOutputBackend().createFile(
      OutputPath, OutputConfig()
                      .setTextWithCRLF(false)
                      .setDiscardOnSignal(true)
                      .setAtomicWrite(true)
                      // To avoid failures that would not happen in uncached
                      // builds, always create the parent directory. See comment
                      // in replayCachedResult.
                      .setImplyCreateDirectories(true));
  if (!O)
    return O.takeError();

  O->discardOnDestroy([](llvm::Error E) { consumeError(std::move(E)); });
  return O;
}

Expected<bool> ObjectStoreCachingOutputs::tryReplayCachedResult(
    const llvm::cas::CASID &ResultCacheKey) {
  DiagnosticsEngine &Diags = Clang.getDiagnostics();

  Optional<llvm::cas::CASID> Result;
  {
    llvm::ScopedDurationTimer ScopedTime([&Diags](double Seconds) {
      Diags.Report(diag::remark_compile_job_cache_timing_backend_key_query)
          << llvm::format("%.6fs", Seconds);
    });
    if (Error E = Cache->get(ResultCacheKey).moveInto(Result))
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

  Optional<llvm::cas::ObjectRef> ResultRef = CAS->getReference(*Result);
  if (!ResultRef) {
    Diags.Report(diag::remark_compile_job_cache_miss_result_not_found)
        << ResultCacheKey.toString() << "result not in CAS";
    return false;
  }

  Diags.Report(diag::remark_compile_job_cache_hit)
      << ResultCacheKey.toString() << CAS->getID(*ResultRef).toString();
  Optional<int> Status =
      replayCachedResult(*ResultRef, /*JustComputedResult=*/false);
  assert(Status && "Expected a status for a cache hit");
  assert(*Status == 0 && "Expected success status for a cache hit");
  return true;
}

Optional<int> CompileJobCache::tryReplayCachedResult(CompilerInstance &Clang) {
  if (!CacheCompileJob)
    return None;

  DiagnosticsEngine &Diags = Clang.getDiagnostics();

  assert(ResultCacheKey.has_value() && "ResultCacheKey not initialized?");

  Clang.setCompileJobCacheKey(*ResultCacheKey);

  Expected<bool> ReplayedResult =
      DisableCachedCompileJobReplay
          ? false
          : CacheBackend->tryReplayCachedResult(*ResultCacheKey);
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

  return None;
}

bool CachingOutputs::prepareOutputCollectionCommon(
    IntrusiveRefCntPtr<llvm::vfs::OutputBackend> CacheOutputs) {
  // Create an on-disk backend for streaming the results live if we run the
  // computation. If we're writing the output as a CASID, skip it here, since
  // it'll be handled during replay.
  IntrusiveRefCntPtr<llvm::vfs::OutputBackend> OnDiskOutputs =
      llvm::makeIntrusiveRefCnt<llvm::vfs::OnDiskOutputBackend>();

  // Set up the output backend so we can save / cache the result after.
  for (OutputKind K : cas::CompileJobCacheResult::getAllOutputKinds()) {
    StringRef OutPath = getPathForOutputKind(K);
    if (!OutPath.empty())
      CachedResultBuilder.addKindMap(K, OutPath);
  }

  // Always filter out the dependencies file, since we build a CAS-specific
  // object for it.
  auto FilterBackend = llvm::vfs::makeFilteringOutputBackend(
      CacheOutputs,
      [&](StringRef Path, Optional<llvm::vfs::OutputConfig> Config) {
        return Path != DependenciesFile;
      });

  Clang.setOutputBackend(llvm::vfs::makeMirroringOutputBackend(
      FilterBackend, std::move(OnDiskOutputs)));
  ResultDiagsOS = std::make_unique<raw_mirroring_ostream>(
      llvm::errs(), std::make_unique<llvm::raw_svector_ostream>(ResultDiags));

  // FIXME: This should be saving/replaying structured diagnostics, not saving
  // stderr and a separate diagnostics file, thus using the current llvm::errs()
  // colour capabilities and making the choice of whether colors are used, or
  // whether a serialized diagnostics file is emitted, not affect the
  // compilation key. We still want to print errors live during this
  // compilation, just also serialize them. Another benefit of saving structured
  // diagnostics is that it will enable remapping canonicalized paths in
  // diagnostics to their non-canical form for displaying purposes
  // (rdar://85234207).
  //
  // Note that the serialized diagnostics file format loses information, e.g.
  // the include stack is written as additional 'note' diagnostics but when
  // printed in terminal the include stack is printed in a different way than
  // 'note' diagnostics. We should serialize/deserialize diagnostics in a way
  // that we can accurately feed them to a DiagnosticConsumer (whatever that
  // consumer implementation is doing). A potential way is to serialize data
  // that can be deserialized as 'StoredDiagnostic's, which would be close to
  // what the DiagnosticConsumers expect.

  // Notify the existing diagnostic client that all files were processed.
  Clang.getDiagnosticClient().finish();

  DiagnosticsEngine &Diags = Clang.getDiagnostics();
  DiagnosticOptions &DiagOpts = Clang.getInvocation().getDiagnosticOpts();
  Clang.getDiagnostics().setClient(
      new TextDiagnosticPrinter(*ResultDiagsOS, &DiagOpts),
      /*ShouldOwnClient=*/true);
  if (!DiagOpts.DiagnosticSerializationFile.empty()) {
    // Save the serialized diagnostics file as CAS output.
    if (Error E =
            createBinaryOutputFile(Clang, DiagOpts.DiagnosticSerializationFile)
                .moveInto(SerialDiagsOutput)) {
      Diags.Report(diag::err_fe_unable_to_open_output)
          << DiagOpts.DiagnosticSerializationFile
          << errorToErrorCode(std::move(E)).message();
      return 1;
    }

    Expected<std::unique_ptr<raw_pwrite_stream>> OS =
        SerialDiagsOutput->createProxy();
    if (!OS) {
      Diags.Report(diag::err_fe_unable_to_open_output)
          << DiagOpts.DiagnosticSerializationFile
          << errorToErrorCode(OS.takeError()).message();
      return 1;
    }
    auto SerializedConsumer = clang::serialized_diags::create(
        OutputFile, &DiagOpts, /*MergeChildRecords*/ false, std::move(*OS));
    Diags.setClient(new ChainedDiagnosticConsumer(
        Diags.takeClient(), std::move(SerializedConsumer)));
  } else {
    // We always generate the serialized diagnostics so the key is independent
    // of the presence of '--serialize-diagnostics'.
    auto OS = std::make_unique<llvm::raw_svector_ostream>(SerialDiagsBuf);
    auto SerializedConsumer = clang::serialized_diags::create(
        StringRef(), &DiagOpts, /*MergeChildRecords*/ false, std::move(OS));
    Diags.setClient(new ChainedDiagnosticConsumer(
        Diags.takeClient(), std::move(SerializedConsumer)));
  }

  return false;
}

bool ObjectStoreCachingOutputs::prepareOutputCollection() {
  if (prepareOutputCollectionCommon(CASOutputs))
    return true;

  if (!Clang.getDependencyOutputOpts().OutputFile.empty())
    Clang.addDependencyCollector(std::make_shared<CASDependencyCollector>(
        Clang.getDependencyOutputOpts(), *CAS,
        [this](Optional<cas::ObjectRef> Deps) { DependenciesOutput = Deps; }));

  return false;
}

bool CompileJobCache::finishComputedResult(CompilerInstance &Clang,
                                           bool Success) {
  // Nothing to do if not caching.
  if (!CacheCompileJob)
    return Success;

  CacheBackend->finishSerializedDiagnostics();

  // Don't cache failed builds.
  //
  // TODO: Consider caching failed builds! Note: when output files are written
  // without a temporary (non-atomically), failure may cause the removal of a
  // preexisting file. That behaviour is not currently modeled by the cache.
  if (!Success)
    return false;

  DiagnosticsEngine &Diags = Clang.getDiagnostics();

  // Existing diagnostic client is finished, create a new one in case we need
  // to print more diagnostics.
  Diags.setClient(new TextDiagnosticPrinter(
                      llvm::errs(), &Clang.getInvocation().getDiagnosticOpts()),
                  /*ShouldOwnClient=*/true);

  // Check if we encounter any source that would generate non-reproducible
  // outputs.
  bool SkipCache = Clang.hasPreprocessor() && Clang.isSourceNonReproducible();
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

  if (Error E =
          CacheBackend->finishComputedResult(*ResultCacheKey, SkipCache)) {
    reportCachingBackendError(Diags, std::move(E));
    return false;
  }
  return true;
}

void CachingOutputs::finishSerializedDiagnostics() {
  if (SerialDiagsOutput) {
    llvm::handleAllErrors(
        SerialDiagsOutput->keep(),
        [&](const llvm::vfs::TempFileOutputError &E) {
          Clang.getDiagnostics().Report(diag::err_unable_to_rename_temp)
              << E.getTempPath() << E.getOutputPath()
              << E.convertToErrorCode().message();
        },
        [&](const llvm::vfs::OutputError &E) {
          Clang.getDiagnostics().Report(diag::err_fe_unable_to_open_output)
              << E.getOutputPath() << E.convertToErrorCode().message();
        });
  }
}

Expected<cas::ObjectRef> ObjectStoreCachingOutputs::writeOutputs(
    const llvm::cas::CASID &ResultCacheKey) {
  DiagnosticsEngine &Diags = Clang.getDiagnostics();
  llvm::ScopedDurationTimer ScopedTime([&Diags](double Seconds) {
    Diags.Report(diag::remark_compile_job_cache_timing_backend_store)
        << llvm::format("%.6fs", Seconds);
  });

  if (!SerialDiagsOutput) {
    // Not requested to get a serialized diagnostics file but we generated it
    // and will store it regardless so that the key is independent of the
    // presence of '--serialize-diagnostics'.
    Expected<llvm::cas::ObjectProxy> SerialDiags =
        CAS->createProxy(None, SerialDiagsBuf);
    if (!SerialDiags)
      return SerialDiags.takeError();
    CachedResultBuilder.addOutput(OutputKind::SerializedDiagnostics,
                                  SerialDiags->getRef());
  }

  if (DependenciesOutput)
    CachedResultBuilder.addOutput(OutputKind::Dependencies,
                                  *DependenciesOutput);

  auto BackendOutputs = CASOutputs->takeOutputs();
  for (auto &Output : BackendOutputs)
    if (auto Err = CachedResultBuilder.addOutput(Output.Path, Output.Object))
      return std::move(Err);

  // Hack around llvm::errs() not being captured by the output backend yet.
  Expected<llvm::cas::ObjectRef> Errs = CAS->storeFromString(None, ResultDiags);
  if (!Errs)
    return Errs.takeError();
  CachedResultBuilder.addOutput(OutputKind::Stderr, *Errs);

  // Cache the result.
  return CachedResultBuilder.build(*CAS);
}

Error ObjectStoreCachingOutputs::finishComputedResult(
    const llvm::cas::CASID &ResultCacheKey, bool SkipCache) {
  Expected<cas::ObjectRef> Result = writeOutputs(ResultCacheKey);
  if (!Result)
    return Result.takeError();

  // Skip caching if requested.
  if (!SkipCache) {
    DiagnosticsEngine &Diags = Clang.getDiagnostics();
    llvm::ScopedDurationTimer ScopedTime([&Diags](double Seconds) {
      Diags.Report(diag::remark_compile_job_cache_timing_backend_key_update)
          << llvm::format("%.6fs", Seconds);
    });
    if (llvm::Error E = Cache->put(ResultCacheKey, CAS->getID(*Result)))
      return E;
  }

  // Replay / decanonicalize as necessary.
  Optional<int> Status = replayCachedResult(*Result,
                                            /*JustComputedResult=*/true);
  (void)Status;
  assert(Status == None);
  return Error::success();
}

/// Replay a result after a cache hit.
Optional<int>
ObjectStoreCachingOutputs::replayCachedResult(llvm::cas::ObjectRef ResultID,
                                              bool JustComputedResult) {
  if (JustComputedResult)
    return None;

  if (!JustComputedResult) {
    // Disable the existing DiagnosticConsumer, we'll both print to stderr
    // directly and also potentially output a serialized diagnostics file, in
    // which case we don't want the outer DiagnosticConsumer to overwrite it and
    // lose the compilation diagnostics.
    // See FIXME in CompileJobCache::tryReplayCachedResult() about improving how
    // we handle diagnostics for caching purposes.
    Clang.getDiagnosticClient().finish();
    DiagnosticOptions &DiagOpts = Clang.getInvocation().getDiagnosticOpts();
    Clang.getDiagnostics().setClient(
        new TextDiagnosticPrinter(llvm::errs(), &DiagOpts),
        /*ShouldOwnClient=*/true);
  }

  // FIXME: Stop calling report_fatal_error().
  Optional<cas::CompileJobCacheResult> Result;
  cas::CompileJobResultSchema Schema(*CAS);
  if (Error E = Schema.load(ResultID).moveInto(Result))
    llvm::report_fatal_error(std::move(E));

  auto Err = Result->forEachOutput([&](cas::CompileJobCacheResult::Output O)
                                       -> Error {
    if (O.Kind == OutputKind::Stderr) {
      Optional<llvm::cas::ObjectProxy> Errs;
      if (Error E = CAS->getProxy(O.Object).moveInto(Errs))
        return E;
      llvm::errs() << Errs->getData();
      return Error::success(); // continue
    }

    std::string Path = std::string(getPathForOutputKind(O.Kind));
    if (Path.empty())
      // The output may be always generated but not needed with this invocation,
      // like the serialized diagnostics file.
      return Error::success(); // continue

    // Always create parent directory of outputs, since it is hard to precisely
    // match which outputs rely on creating parents and the order outputs are
    // replayed in, in case a previous output would create the parent
    // (e.g. a .pcm and .diag file in the same directory).
    StringRef ParentPath = llvm::sys::path::parent_path(Path);
    if (!ParentPath.empty())
      llvm::sys::fs::create_directories(ParentPath);

    Optional<StringRef> Contents;
    SmallString<50> ContentsStorage;
    if (O.Kind == OutputKind::Dependencies) {
      llvm::raw_svector_ostream OS(ContentsStorage);
      if (auto E = CASDependencyCollector::replay(
              Clang.getDependencyOutputOpts(), *CAS, O.Object, OS))
        return E;
      Contents = ContentsStorage;
    } else {
      Optional<llvm::cas::ObjectProxy> Bytes;
      if (Error E = CAS->getProxy(O.Object).moveInto(Bytes))
        return E;
      Contents = Bytes->getData();
    }

    std::unique_ptr<llvm::FileOutputBuffer> Output;
    if (Error E = llvm::FileOutputBuffer::create(Path, Contents->size())
                      .moveInto(Output))
      return E;
    llvm::copy(*Contents, Output->getBufferStart());
    return Output->commit();
  });

  // FIXME: Stop calling report_fatal_error().
  if (Err)
    llvm::report_fatal_error(std::move(Err));

  if (JustComputedResult)
    return None;
  return 0;
}

Expected<bool> RemoteCachingOutputs::tryReplayCachedResult(
    const llvm::cas::CASID &ResultCacheKey) {
  DiagnosticsEngine &Diags = Clang.getDiagnostics();

  Optional<llvm::cas::remote::KeyValueDBClient::GetValueAsyncQueue::Response>
      Response;
  {
    llvm::ScopedDurationTimer ScopedTime([&Diags](double Seconds) {
      Diags.Report(diag::remark_compile_job_cache_timing_backend_key_query)
          << llvm::format("%.6fs", Seconds);
    });
    RemoteKVClient->getValueQueue().getValueAsync(ResultCacheKey.getHash());
    if (Error E =
            RemoteKVClient->getValueQueue().receiveNext().moveInto(Response))
      return std::move(E);
  }
  if (!Response->Value) {
    Diags.Report(diag::remark_compile_job_cache_miss)
        << ResultCacheKey.toString();
    return false;
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
static constexpr llvm::StringLiteral StderrDiagnosticsKindName =
    "<stderr-diags>";
static constexpr llvm::StringLiteral DependenciesOutputKindName =
    "<dependencies>";

StringRef RemoteCachingOutputs::getOutputKindName(OutputKind Kind) {
  switch (Kind) {
  case OutputKind::MainOutput:
    return MainOutputKindName;
  case OutputKind::SerializedDiagnostics:
    return SerializedDiagnosticsKindName;
  case OutputKind::Stderr:
    return StderrDiagnosticsKindName;
  case OutputKind::Dependencies:
    return DependenciesOutputKindName;
  }
}

Optional<CachingOutputs::OutputKind>
RemoteCachingOutputs::getOutputKindForName(StringRef Name) {
  return llvm::StringSwitch<Optional<OutputKind>>(Name)
      .Case(MainOutputKindName, OutputKind::MainOutput)
      .Case(SerializedDiagnosticsKindName, OutputKind::SerializedDiagnostics)
      .Case(StderrDiagnosticsKindName, OutputKind::Stderr)
      .Case(DependenciesOutputKindName, OutputKind::Dependencies)
      .Default(None);
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

  // Disable the existing DiagnosticConsumer, we'll both print to stderr
  // directly and also potentially output a serialized diagnostics file, in
  // which case we don't want the outer DiagnosticConsumer to overwrite it and
  // lose the compilation diagnostics.
  // See FIXME in CompileJobCache::tryReplayCachedResult() about improving how
  // we handle diagnostics for caching purposes.
  Clang.getDiagnosticClient().finish();
  // Insert a text printer so that the caching remarks can be printed.
  DiagnosticsEngine &Diags = Clang.getDiagnostics();
  Diags.setClient(new TextDiagnosticPrinter(
                      llvm::errs(), &Clang.getInvocation().getDiagnosticOpts()),
                  /*ShouldOwnClient=*/true);

  // Replay outputs.

  auto &LoadQueue = RemoteCASClient->loadQueue();
  struct CallCtx : public llvm::cas::remote::AsyncCallerContext {
    StringRef OutputName;
    StringRef CASID;
    bool IsStderr;
    CallCtx(StringRef OutputName, StringRef CASID, bool IsStderr)
        : OutputName(OutputName), CASID(CASID), IsStderr(IsStderr) {}
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

    Optional<OutputKind> OutKind = getOutputKindForName(OutputName);
    StringRef Path = OutKind ? getPathForOutputKind(*OutKind) : OutputName;

    if (OutKind && *OutKind == OutputKind::Stderr) {
      LoadQueue.loadAsync(CASID, /*OutFilePath*/ None,
                          makeCtx(OutputName, CASID, /*IsStderr*/ true));
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
  Optional<std::string> StderrDiags;

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

    if (Ctx.IsStderr)
      StderrDiags = std::move(Response->BlobData);
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

  if (StderrDiags)
    llvm::errs() << *StderrDiags;

  return true;
}

bool RemoteCachingOutputs::prepareOutputCollection() {
  // Set up the output backend so we can save / cache the result after.
  for (OutputKind K : cas::CompileJobCacheResult::getAllOutputKinds()) {
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

  if (!SerialDiagsOutput) {
    // Not requested to get a serialized diagnostics file but we generated it
    // and will store it regardless so that the key is independent of the
    // presence of '--serialize-diagnostics'.
    SaveQueue.saveDataAsync(
        SerialDiagsBuf.str().str(),
        makeCtx(getOutputKindName(OutputKind::SerializedDiagnostics)));
  }

  // FIXME: Save dependencies output.

  if (!ResultDiags.empty()) {
    SaveQueue.saveDataAsync(ResultDiags.str().str(),
                            makeCtx(getOutputKindName(OutputKind::Stderr)));
  }

  for (StringRef OutputName : CollectingOutputs->getOutputs()) {
    Optional<OutputKind> OutKind = getOutputKindForName(OutputName);
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

Error RemoteCachingOutputs::finishComputedResult(
    const llvm::cas::CASID &ResultCacheKey, bool SkipCache) {
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

  RemoteKVClient->putValueQueue().putValueAsync(ResultCacheKey.getHash(),
                                                *CompResult);
  auto Response = RemoteKVClient->putValueQueue().receiveNext();
  if (!Response)
    return Response.takeError();

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
  if (const char *LLTaskID = getenv("LLBUILD_TASK_ID")) {
    // Use the llbuild protocol to request to release the execution lane for
    // this task.
    const char *LLControlFD = getenv("LLBUILD_CONTROL_FD");
    if (!LLControlFD)
      llvm::report_fatal_error("'LLBUILD_CONTROL_FD' env var is not set!");
    int LLCtrlFD;
    bool HasErr = StringRef(LLControlFD).getAsInteger(10, LLCtrlFD);
    if (HasErr)
      llvm::report_fatal_error(Twine("failed converting 'LLBUILD_CONTROL_FD' "
                                     "to an integer, it was: ") +
                               LLControlFD);
    llvm::raw_fd_ostream FDOS(LLCtrlFD, /*shouldClose*/ false);
    FDOS << "llbuild.1\n" << LLTaskID << '\n';
    FDOS.flush();
  }
}

int cc1_main(ArrayRef<const char *> Argv, const char *Argv0, void *MainAddr) {
  ensureSufficientStack();

  CompileJobCache JobCache;
  std::unique_ptr<CompilerInstance> Clang(new CompilerInstance());
  IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());

  // Register the support for object-file-wrapped Clang modules.
  auto PCHOps = Clang->getPCHContainerOperations();
  PCHOps->registerWriter(std::make_unique<ObjectFilePCHContainerWriter>());
  PCHOps->registerReader(std::make_unique<ObjectFilePCHContainerReader>());

  // Initialize targets first, so that --version shows registered targets.
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeAllAsmParsers();

  // Buffer diagnostics from argument parsing so that we can output them using a
  // well formed diagnostic object.
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();
  TextDiagnosticBuffer *DiagsBuffer = new TextDiagnosticBuffer;
  DiagnosticsEngine Diags(DiagID, &*DiagOpts, DiagsBuffer);

  // Setup round-trip remarks for the DiagnosticsEngine used in CreateFromArgs.
  if (find(Argv, StringRef("-Rround-trip-cc1-args")) != Argv.end())
    Diags.setSeverity(diag::remark_cc1_round_trip_generated,
                      diag::Severity::Remark, {});

  bool Success = CompilerInvocation::CreateFromArgs(Clang->getInvocation(),
                                                    Argv, Diags, Argv0);

  if (Clang->getFrontendOpts().TimeTrace ||
      !Clang->getFrontendOpts().TimeTracePath.empty()) {
    Clang->getFrontendOpts().TimeTrace = 1;
    llvm::timeTraceProfilerInitialize(
        Clang->getFrontendOpts().TimeTraceGranularity, Argv0);
  }
  // --print-supported-cpus takes priority over the actual compilation.
  if (Clang->getFrontendOpts().PrintSupportedCPUs)
    return PrintSupportedCPUs(Clang->getTargetOpts().Triple);

  // Infer the builtin include path if unspecified.
  if (Clang->getHeaderSearchOpts().UseBuiltinIncludes &&
      Clang->getHeaderSearchOpts().ResourceDir.empty())
    Clang->getHeaderSearchOpts().ResourceDir =
      CompilerInvocation::GetResourcesPath(Argv0, MainAddr);

  // Create the actual diagnostics engine.
  Clang->createDiagnostics();
  if (!Clang->hasDiagnostics())
    return 1;

  // Set an error handler, so that any LLVM backend diagnostics go through our
  // error handler.
  llvm::install_fatal_error_handler(LLVMErrorHandler,
                                  static_cast<void*>(&Clang->getDiagnostics()));

  DiagsBuffer->FlushDiagnostics(Clang->getDiagnostics());

  auto FinishDiagnosticClient = llvm::make_scope_exit([&]() {
    // Notify the diagnostic client that all files were processed.
    Clang->getDiagnosticClient().finish();

    // Our error handler depends on the Diagnostics object, which we're
    // potentially about to delete. Uninstall the handler now so that any
    // later errors use the default handling behavior instead.
    llvm::remove_fatal_error_handler();
  });

  if (!Success)
    return 1;

  // Initialize caching and replay, if enabled.
  if (Optional<int> Status = JobCache.initialize(*Clang))
    return *Status; // FIXME: Should write out timers before exiting!

  // Check for a cache hit.
  if (Optional<int> Status = JobCache.tryReplayCachedResult(*Clang))
    return *Status; // FIXME: Should write out timers before exiting!

  // ExecuteAction takes responsibility.
  FinishDiagnosticClient.release();

  // Execute the frontend actions.
  {
    llvm::TimeTraceScope TimeScope("ExecuteCompiler");
    Success = ExecuteCompilerInvocation(Clang.get());
  }

  // Cache the result, and decanonicalize and finish outputs.
  Success = JobCache.finishComputedResult(*Clang, Success);

  // If any timers were active but haven't been destroyed yet, print their
  // results now.  This happens in -disable-free mode.
  llvm::TimerGroup::printAll(llvm::errs());
  llvm::TimerGroup::clearAll();

  if (llvm::timeTraceProfilerEnabled()) {
    SmallString<128> Path(Clang->getFrontendOpts().OutputFile);
    llvm::sys::path::replace_extension(Path, "json");
    if (!Clang->getFrontendOpts().TimeTracePath.empty()) {
      // replace the suffix to '.json' directly
      SmallString<128> TracePath(Clang->getFrontendOpts().TimeTracePath);
      if (llvm::sys::fs::is_directory(TracePath))
        llvm::sys::path::append(TracePath, llvm::sys::path::filename(Path));
      Path.assign(TracePath);
    }
    llvm::vfs::OnDiskOutputBackend Backend;
    if (Optional<llvm::vfs::OutputFile> profilerOutput =
            llvm::expectedToOptional(
                Backend.createFile(Path, llvm::vfs::OutputConfig()
                                                  .setTextWithCRLF()
                                                  .setNoDiscardOnSignal()
                                                  .setNoAtomicWrite()))) {
      llvm::timeTraceProfilerWrite(*profilerOutput);
      llvm::consumeError(profilerOutput->keep());
      llvm::timeTraceProfilerCleanup();
    }
  }

  // When running with -disable-free, don't do any destruction or shutdown.
  if (Clang->getFrontendOpts().DisableFree) {
    llvm::BuryPointer(std::move(Clang));
    return !Success;
  }

  return !Success;
}
