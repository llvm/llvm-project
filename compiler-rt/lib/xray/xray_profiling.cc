//===-- xray_profiling.cc ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of XRay, a dynamic runtime instrumentation system.
//
// This is the implementation of a profiling handler.
//
//===----------------------------------------------------------------------===//
#include <memory>
#include <time.h>

#include "sanitizer_common/sanitizer_atomic.h"
#include "sanitizer_common/sanitizer_flags.h"
#include "xray/xray_interface.h"
#include "xray/xray_log_interface.h"

#include "xray_flags.h"
#include "xray_profile_collector.h"
#include "xray_profiling_flags.h"
#include "xray_recursion_guard.h"
#include "xray_tsc.h"
#include "xray_utils.h"
#include <pthread.h>

namespace __xray {

namespace {

atomic_sint32_t ProfilerLogFlushStatus = {
    XRayLogFlushStatus::XRAY_LOG_NOT_FLUSHING};

atomic_sint32_t ProfilerLogStatus = {XRayLogInitStatus::XRAY_LOG_UNINITIALIZED};

SpinMutex ProfilerOptionsMutex;

struct alignas(64) ProfilingData {
  FunctionCallTrie::Allocators *Allocators;
  FunctionCallTrie *FCT;
};

static pthread_key_t ProfilingKey;

thread_local std::aligned_storage<sizeof(FunctionCallTrie::Allocators)>::type
    AllocatorsStorage;
thread_local std::aligned_storage<sizeof(FunctionCallTrie)>::type
    FunctionCallTrieStorage;
thread_local std::aligned_storage<sizeof(ProfilingData)>::type ThreadStorage{};

static ProfilingData &getThreadLocalData() XRAY_NEVER_INSTRUMENT {
  thread_local auto ThreadOnce = [] {
    new (&ThreadStorage) ProfilingData{};
    auto *Allocators =
        reinterpret_cast<FunctionCallTrie::Allocators *>(&AllocatorsStorage);
    new (Allocators) FunctionCallTrie::Allocators();
    *Allocators = FunctionCallTrie::InitAllocators();
    auto *FCT = reinterpret_cast<FunctionCallTrie *>(&FunctionCallTrieStorage);
    new (FCT) FunctionCallTrie(*Allocators);
    auto &TLD = *reinterpret_cast<ProfilingData *>(&ThreadStorage);
    TLD.Allocators = Allocators;
    TLD.FCT = FCT;
    pthread_setspecific(ProfilingKey, &ThreadStorage);
    return false;
  }();
  (void)ThreadOnce;

  auto &TLD = *reinterpret_cast<ProfilingData*>(&ThreadStorage);

  if (UNLIKELY(TLD.Allocators == nullptr || TLD.FCT == nullptr)) {
    auto *Allocators =
        reinterpret_cast<FunctionCallTrie::Allocators *>(&AllocatorsStorage);
    new (Allocators) FunctionCallTrie::Allocators();
    *Allocators = FunctionCallTrie::InitAllocators();
    auto *FCT = reinterpret_cast<FunctionCallTrie *>(&FunctionCallTrieStorage);
    new (FCT) FunctionCallTrie(*Allocators);
    TLD.Allocators = Allocators;
    TLD.FCT = FCT;
  }

  return *reinterpret_cast<ProfilingData *>(&ThreadStorage);
}

static void cleanupTLD() XRAY_NEVER_INSTRUMENT {
  auto &TLD = *reinterpret_cast<ProfilingData *>(&ThreadStorage);
  if (TLD.Allocators != nullptr && TLD.FCT != nullptr) {
    TLD.FCT->~FunctionCallTrie();
    TLD.Allocators->~Allocators();
    TLD.FCT = nullptr;
    TLD.Allocators = nullptr;
  }
}

} // namespace

const char *profilingCompilerDefinedFlags() XRAY_NEVER_INSTRUMENT {
#ifdef XRAY_PROFILER_DEFAULT_OPTIONS
  return SANITIZER_STRINGIFY(XRAY_PROFILER_DEFAULT_OPTIONS);
#else
  return "";
#endif
}

atomic_sint32_t ProfileFlushStatus = {
    XRayLogFlushStatus::XRAY_LOG_NOT_FLUSHING};

XRayLogFlushStatus profilingFlush() XRAY_NEVER_INSTRUMENT {
  if (atomic_load(&ProfilerLogStatus, memory_order_acquire) !=
      XRayLogInitStatus::XRAY_LOG_FINALIZED) {
    if (Verbosity())
      Report("Not flushing profiles, profiling not been finalized.\n");
    return XRayLogFlushStatus::XRAY_LOG_NOT_FLUSHING;
  }

  s32 Result = XRayLogFlushStatus::XRAY_LOG_NOT_FLUSHING;
  if (!atomic_compare_exchange_strong(&ProfilerLogFlushStatus, &Result,
                                      XRayLogFlushStatus::XRAY_LOG_FLUSHING,
                                      memory_order_acq_rel)) {
    if (Verbosity())
      Report("Not flushing profiles, implementation still finalizing.\n");
  }

  // At this point, we'll create the file that will contain the profile, but
  // only if the options say so.
  if (!profilingFlags()->no_flush) {
    // First check whether we have data in the profile collector service
    // before we try and write anything down.
    XRayBuffer B = profileCollectorService::nextBuffer({nullptr, 0});
    if (B.Data == nullptr) {
      if (Verbosity())
        Report("profiling: No data to flush.\n");
    } else {
      int Fd = getLogFD();
      if (Fd == -1) {
        if (Verbosity())
          Report("profiling: Failed to flush to file, dropping data.\n");
      } else {
        // Now for each of the buffers, write out the profile data as we would
        // see it in memory, verbatim.
        while (B.Data != nullptr && B.Size != 0) {
          retryingWriteAll(Fd, reinterpret_cast<const char *>(B.Data),
                           reinterpret_cast<const char *>(B.Data) + B.Size);
          B = profileCollectorService::nextBuffer(B);
        }
        // Then we close out the file.
        internal_close(Fd);
      }
    }
  }

  profileCollectorService::reset();

  // Flush the current thread's local data structures as well.
  cleanupTLD();

  atomic_store(&ProfilerLogStatus, XRayLogFlushStatus::XRAY_LOG_FLUSHED,
               memory_order_release);

  return XRayLogFlushStatus::XRAY_LOG_FLUSHED;
}

namespace {

thread_local atomic_uint8_t ReentranceGuard{0};

static void postCurrentThreadFCT(ProfilingData &TLD) {
  if (TLD.Allocators == nullptr || TLD.FCT == nullptr)
    return;

  profileCollectorService::post(*TLD.FCT, GetTid());
  cleanupTLD();
}

} // namespace

void profilingHandleArg0(int32_t FuncId,
                         XRayEntryType Entry) XRAY_NEVER_INSTRUMENT {
  unsigned char CPU;
  auto TSC = readTSC(CPU);
  RecursionGuard G(ReentranceGuard);
  if (!G)
    return;

  auto Status = atomic_load(&ProfilerLogStatus, memory_order_acquire);
  if (UNLIKELY(Status == XRayLogInitStatus::XRAY_LOG_FINALIZED ||
               Status == XRayLogInitStatus::XRAY_LOG_FINALIZING)) {
    auto &TLD = getThreadLocalData();
    postCurrentThreadFCT(TLD);
    return;
  }

  auto &TLD = getThreadLocalData();
  switch (Entry) {
  case XRayEntryType::ENTRY:
  case XRayEntryType::LOG_ARGS_ENTRY:
    TLD.FCT->enterFunction(FuncId, TSC);
    break;
  case XRayEntryType::EXIT:
  case XRayEntryType::TAIL:
    TLD.FCT->exitFunction(FuncId, TSC);
    break;
  default:
    // FIXME: Handle bugs.
    break;
  }
}

void profilingHandleArg1(int32_t FuncId, XRayEntryType Entry,
                         uint64_t) XRAY_NEVER_INSTRUMENT {
  return profilingHandleArg0(FuncId, Entry);
}

XRayLogInitStatus profilingFinalize() XRAY_NEVER_INSTRUMENT {
  s32 CurrentStatus = XRayLogInitStatus::XRAY_LOG_INITIALIZED;
  if (!atomic_compare_exchange_strong(&ProfilerLogStatus, &CurrentStatus,
                                      XRayLogInitStatus::XRAY_LOG_FINALIZING,
                                      memory_order_release)) {
    if (Verbosity())
      Report("Cannot finalize profile, the profiling is not initialized.\n");
    return static_cast<XRayLogInitStatus>(CurrentStatus);
  }

  // Wait a grace period to allow threads to see that we're finalizing.
  SleepForMillis(profilingFlags()->grace_period_ms);

  // We also want to make sure that the current thread's data is cleaned up,
  // if we have any.
  auto &TLD = getThreadLocalData();
  postCurrentThreadFCT(TLD);

  // Then we force serialize the log data.
  profileCollectorService::serialize();

  atomic_store(&ProfilerLogStatus, XRayLogInitStatus::XRAY_LOG_FINALIZED,
               memory_order_release);
  return XRayLogInitStatus::XRAY_LOG_FINALIZED;
}

XRayLogInitStatus
profilingLoggingInit(size_t BufferSize, size_t BufferMax, void *Options,
                     size_t OptionsSize) XRAY_NEVER_INSTRUMENT {
  if (BufferSize != 0 || BufferMax != 0) {
    if (Verbosity())
      Report("__xray_log_init() being used, and is unsupported. Use "
             "__xray_log_init_mode(...) instead. Bailing out.");
    return XRayLogInitStatus::XRAY_LOG_UNINITIALIZED;
  }

  s32 CurrentStatus = XRayLogInitStatus::XRAY_LOG_UNINITIALIZED;
  if (!atomic_compare_exchange_strong(&ProfilerLogStatus, &CurrentStatus,
                                      XRayLogInitStatus::XRAY_LOG_INITIALIZING,
                                      memory_order_release)) {
    if (Verbosity())
      Report("Cannot initialize already initialised profiling "
             "implementation.\n");
    return static_cast<XRayLogInitStatus>(CurrentStatus);
  }

  {
    SpinMutexLock Lock(&ProfilerOptionsMutex);
    FlagParser ConfigParser;
    ProfilerFlags Flags;
    Flags.setDefaults();
    registerProfilerFlags(&ConfigParser, &Flags);
    ConfigParser.ParseString(profilingCompilerDefinedFlags());
    const char *Env = GetEnv("XRAY_PROFILING_OPTIONS");
    if (Env == nullptr)
      Env = "";
    ConfigParser.ParseString(Env);

    // Then parse the configuration string provided.
    ConfigParser.ParseString(static_cast<const char *>(Options));
    if (Verbosity())
      ReportUnrecognizedFlags();
    *profilingFlags() = Flags;
  }

  // We need to reset the profile data collection implementation now.
  profileCollectorService::reset();

  // We need to set up the exit handlers.
  static pthread_once_t Once = PTHREAD_ONCE_INIT;
  pthread_once(&Once, +[] {
    pthread_key_create(&ProfilingKey, +[](void *P) {
      // This is the thread-exit handler.
      auto &TLD = *reinterpret_cast<ProfilingData *>(P);
      if (TLD.Allocators == nullptr && TLD.FCT == nullptr)
        return;

      postCurrentThreadFCT(TLD);
    });

    // We also need to set up an exit handler, so that we can get the profile
    // information at exit time. We use the C API to do this, to not rely on C++
    // ABI functions for registering exit handlers.
    Atexit(+[] {
      // Finalize and flush.
      if (profilingFinalize() != XRAY_LOG_FINALIZED) {
        cleanupTLD();
        return;
      }
      if (profilingFlush() != XRAY_LOG_FLUSHED) {
        cleanupTLD();
        return;
      }
      if (Verbosity())
        Report("XRay Profile flushed at exit.");
    });
  });

  __xray_log_set_buffer_iterator(profileCollectorService::nextBuffer);
  __xray_set_handler(profilingHandleArg0);
  __xray_set_handler_arg1(profilingHandleArg1);

  atomic_store(&ProfilerLogStatus, XRayLogInitStatus::XRAY_LOG_INITIALIZED,
               memory_order_release);
  if (Verbosity())
    Report("XRay Profiling init successful.\n");

  return XRayLogInitStatus::XRAY_LOG_INITIALIZED;
}

bool profilingDynamicInitializer() XRAY_NEVER_INSTRUMENT {
  // Set up the flag defaults from the static defaults and the
  // compiler-provided defaults.
  {
    SpinMutexLock Lock(&ProfilerOptionsMutex);
    auto *F = profilingFlags();
    F->setDefaults();
    FlagParser ProfilingParser;
    registerProfilerFlags(&ProfilingParser, F);
    ProfilingParser.ParseString(profilingCompilerDefinedFlags());
  }

  XRayLogImpl Impl{
      profilingLoggingInit,
      profilingFinalize,
      profilingHandleArg0,
      profilingFlush,
  };
  auto RegistrationResult = __xray_log_register_mode("xray-profiling", Impl);
  if (RegistrationResult != XRayLogRegisterStatus::XRAY_REGISTRATION_OK) {
    if (Verbosity())
      Report("Cannot register XRay Profiling mode to 'xray-profiling'; error = "
             "%d\n",
             RegistrationResult);
    return false;
  }

  if (!internal_strcmp(flags()->xray_mode, "xray-profiling"))
    __xray_log_select_mode("xray_profiling");
  return true;
}

} // namespace __xray

static auto UNUSED Unused = __xray::profilingDynamicInitializer();
