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
#include "clang/Frontend/CompileJobCacheKey.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Frontend/TextDiagnosticBuffer.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Frontend/Utils.h"
#include "clang/FrontendTool/Utils.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CAS/CASDB.h"
#include "llvm/CAS/CASFileSystem.h"
#include "llvm/CAS/CASOutputBackend.h"
#include "llvm/CAS/HierarchicalTreeBuilder.h"
#include "llvm/CAS/TreeSchema.h"
#include "llvm/CAS/Utils.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/LinkAllPasses.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Support/BuryPointer.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileOutputBuffer.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
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
  /// Canonicalize \p Clang.
  ///
  /// Return status if should exit immediately, otherwise None.
  ///
  /// TODO: Refactor \a cc1_main() so that instead this canonicalizes the
  /// CompilerInvocation before Clang gets access to command-line arguments, to
  /// control what might leak.
  Optional<int> initialize(CompilerInstance &Clang);

  /// Try looking up a cached result and replaying it.
  ///
  /// Return status if should exit immediately, otherwise None.
  Optional<int> tryReplayCachedResult(CompilerInstance &Clang);

  /// Finish writing outputs from a computed result, after a cache miss.
  void finishComputedResult(CompilerInstance &Clang, bool Success);

private:
  /// Replay a cache hit.
  ///
  /// Return status if should exit immediately, otherwise None.
  Optional<int> replayCachedResult(llvm::cas::ObjectRef ResultID,
                                   bool JustComputedResult);

  bool CacheCompileJob = false;

  std::shared_ptr<llvm::cas::CASDB> CAS;
  SmallString<256> ResultDiags;
  Optional<llvm::cas::CASID> ResultCacheKey;
  std::unique_ptr<llvm::raw_ostream> ResultDiagsOS;
  IntrusiveRefCntPtr<llvm::cas::CASOutputBackend> CASOutputs;
  std::string OutputFile;
};
} // end anonymous namespace

Optional<int> CompileJobCache::initialize(CompilerInstance &Clang) {
  CompilerInvocation &Invocation = Clang.getInvocation();
  DiagnosticsEngine &Diags = Clang.getDiagnostics();

  // Extract whether caching is on (and canonicalize setting).
  CacheCompileJob = Invocation.getFrontendOpts().CacheCompileJob;
  Invocation.getFrontendOpts().CacheCompileJob = false;

  // Nothing else to do if we're not caching.
  if (!CacheCompileJob)
    return None;

  // Hide the CAS configuration, canonicalizing it to keep the path to the
  // CAS from leaking to the compile job, where it might affecting its
  // output (e.g., in a diagnostic).
  //
  // TODO: Extract CASOptions.Path first if we need it later since it'll
  // disappear here.
  CAS = Invocation.getCASOpts().getOrCreateCASAndHideConfig(Diags);
  if (!CAS)
    return 1; // Exit with error!

  // Canonicalize Invocation and save things in a side channel.
  //
  // TODO: Canonicalize DiagnosticOptions here to be "serialized" only. Pass in
  // a hook to mirror diagnostics to stderr (when writing there), and handle
  // other outputs during replay.

  // TODO: Canonicalize OutputFile to "-" here. During replay, move it and
  // derived outputs to the right place.
  OutputFile = Invocation.getFrontendOpts().OutputFile;
  return None;
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

Optional<int> CompileJobCache::tryReplayCachedResult(CompilerInstance &Clang) {
  if (!CacheCompileJob)
    return None;

  // Create the result cache key once Invocation has been canonicalized.
  ResultCacheKey = createCompileJobCacheKey(*CAS, Clang.getDiagnostics(),
                                            Clang.getInvocation());
  if (!ResultCacheKey)
    return 1;

  Expected<llvm::cas::CASID> Result = CAS->getCachedResult(*ResultCacheKey);
  if (Result) {
    if (Optional<llvm::cas::ObjectRef> ResultRef = CAS->getReference(*Result)) {
      Clang.getDiagnostics().Report(diag::remark_compile_job_cache_hit)
          << ResultCacheKey->toString() << Result->toString();
      Optional<int> Status =
          replayCachedResult(*ResultRef, /*JustComputedResult=*/false);
      assert(Status && "Expected a status for a cache hit");
      return *Status;
    }
    Clang.getDiagnostics().Report(
        diag::remark_compile_job_cache_miss_result_not_found)
        << ResultCacheKey->toString() << Result->toString();
  } else {
    llvm::consumeError(Result.takeError());
    Clang.getDiagnostics().Report(diag::remark_compile_job_cache_miss)
        << ResultCacheKey->toString();
  }

  // Create an on-disk backend for streaming the results live if we run the
  // computation. If we're writing the output as a CASID, skip it here, since
  // it'll be handled during replay.
  IntrusiveRefCntPtr<llvm::vfs::OutputBackend> OnDiskOutputs =
      llvm::makeIntrusiveRefCnt<llvm::vfs::OnDiskOutputBackend>();

  // Set up the output backend so we can save / cache the result after.
  CASOutputs = llvm::makeIntrusiveRefCnt<llvm::cas::CASOutputBackend>(*CAS);

  Clang.setOutputBackend(llvm::vfs::makeMirroringOutputBackend(
      CASOutputs, std::move(OnDiskOutputs)));
  ResultDiagsOS = std::make_unique<raw_mirroring_ostream>(
      llvm::errs(), std::make_unique<llvm::raw_svector_ostream>(ResultDiags));

  // FIXME: This should be saving/replaying serialized diagnostics, thus
  // using the current llvm::errs() colour capabilities. We still want to
  // print errors live during this compilation, just also serialize them.
  //
  // However, serialized diagnostics can only be written to a file, not to a
  // raw_ostream. Need to fix that first. Also, maybe the format doesn't need
  // to be bitcode... we just want to serialize them faithfully such that we
  // can decide at output time whether to make the colours pretty.

  // Notify the existing diagnostic client that all files were processed.
  Clang.getDiagnosticClient().finish();

  Clang.getDiagnostics().setClient(
      new TextDiagnosticPrinter(*ResultDiagsOS,
                                &Clang.getInvocation().getDiagnosticOpts()),
      /*ShouldOwnClient=*/true);
  return None;
}

void CompileJobCache::finishComputedResult(CompilerInstance &Clang,
                                           bool Success) {
  // Nothing to do if not caching.
  if (!CacheCompileJob)
    return;

  // Don't cache failed builds.
  //
  // TODO: Consider caching failed builds! Note: when output files are written
  // without a temporary (non-atomically), failure may cause the removal of a
  // preexisting file. That behaviour is not currently modeled by the cache.
  if (!Success)
    return;

  // FIXME: Stop calling report_fatal_error().
  Expected<llvm::cas::ObjectProxy> Outputs = CASOutputs->getCASProxy();
  if (!Outputs)
    llvm::report_fatal_error(Outputs.takeError());

  // Hack around llvm::errs() not being captured by the output backend yet.
  //
  // FIXME: Stop calling report_fatal_error().
  Expected<llvm::cas::ObjectProxy> Errs = CAS->createProxy(None, ResultDiags);
  if (!Errs)
    llvm::report_fatal_error(Errs.takeError());

  // Cache the result.
  //
  // FIXME: Stop calling report_fatal_error().
  llvm::cas::HierarchicalTreeBuilder Builder;
  Builder.push(Outputs->getRef(), llvm::cas::TreeEntry::Regular, "outputs");
  Builder.push(Errs->getRef(), llvm::cas::TreeEntry::Regular, "stderr");
  Expected<llvm::cas::ObjectHandle> Result = Builder.create(*CAS);
  if (!Result)
    llvm::report_fatal_error(Result.takeError());
  if (llvm::Error E =
          CAS->putCachedResult(*ResultCacheKey, CAS->getID(*Result)))
    llvm::report_fatal_error(std::move(E));

  // Replay / decanonicalize as necessary.
  Optional<int> Status = replayCachedResult(CAS->getReference(*Result),
                                            /*JustComputedResult=*/true);
  (void)Status;
  assert(Status == None);
}

/// Replay a result after a cache hit.
Optional<int> CompileJobCache::replayCachedResult(llvm::cas::ObjectRef ResultID,
                                                  bool JustComputedResult) {
  if (JustComputedResult)
    return None;

  // FIXME: Stop calling report_fatal_error().
  Optional<llvm::cas::TreeProxy> Result;
  llvm::cas::TreeSchema Schema(*CAS);
  if (Error E = Schema.load(ResultID).moveInto(Result))
    llvm::report_fatal_error(std::move(E));

  // Replay diagnostics to stderr.
  if (!JustComputedResult) {
    Optional<llvm::cas::ObjectProxy> Errs;
    if (Optional<llvm::cas::NamedTreeEntry> Entry = Result->lookup("stderr"))
      if (Error E = CAS->getProxy(Entry->getRef()).moveInto(Errs))
        llvm::report_fatal_error(std::move(E));
    if (!Errs)
      llvm::report_fatal_error("CAS error accessing stderr");
    llvm::errs() << Errs->getData();
  }

  // Replay outputs.
  //
  // FIXME: Use a NodeReader here once it exists.
  Optional<llvm::cas::ObjectProxy> Outputs;
  if (Optional<llvm::cas::NamedTreeEntry> Entry = Result->lookup("outputs"))
    if (Error E = CAS->getProxy(Entry->getRef()).moveInto(Outputs))
      llvm::report_fatal_error(std::move(E));
  if (!Outputs)
    llvm::report_fatal_error("CAS error accessing outputs");

  for (size_t I = 0, E = Outputs->getNumReferences(); I + 1 < E; I += 2) {
    llvm::cas::CASID PathID = Outputs->getReferenceID(I);
    llvm::cas::CASID BytesID = Outputs->getReferenceID(I + 1);

    Optional<llvm::cas::ObjectProxy> Path;
    if (Error E = CAS->getProxy(PathID).moveInto(Path))
      llvm::report_fatal_error(std::move(E));

    Optional<StringRef> Contents;
    SmallString<50> ContentsStorage;
    Optional<llvm::cas::ObjectProxy> Bytes;
    if (Error E = CAS->getProxy(BytesID).moveInto(Bytes))
      llvm::report_fatal_error(std::move(E));
    Contents = Bytes->getData();
    std::unique_ptr<llvm::FileOutputBuffer> Output;
    if (Error E =
            llvm::FileOutputBuffer::create(Path->getData(), Contents->size())
                .moveInto(Output))
      llvm::report_fatal_error(std::move(E));
    llvm::copy(*Contents, Output->getBufferStart());
    if (llvm::Error E = Output->commit())
      llvm::report_fatal_error(std::move(E));
  }

  if (JustComputedResult)
    return None;
  return 0;
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

  if (Clang->getFrontendOpts().TimeTrace) {
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
  JobCache.finishComputedResult(*Clang, Success);

  // If any timers were active but haven't been destroyed yet, print their
  // results now.  This happens in -disable-free mode.
  llvm::TimerGroup::printAll(llvm::errs());
  llvm::TimerGroup::clearAll();

  if (llvm::timeTraceProfilerEnabled()) {
    SmallString<128> Path(Clang->getFrontendOpts().OutputFile);
    llvm::sys::path::replace_extension(Path, "json");
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
