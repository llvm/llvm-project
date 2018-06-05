//===-- xray_basic_logging.cc -----------------------------------*- C++ -*-===//
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
// Implementation of a simple in-memory log of XRay events. This defines a
// logging function that's compatible with the XRay handler interface, and
// routines for exporting data to files.
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <cstring>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include "sanitizer_common/sanitizer_allocator_internal.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "xray/xray_records.h"
#include "xray_basic_flags.h"
#include "xray_basic_logging.h"
#include "xray_defs.h"
#include "xray_flags.h"
#include "xray_interface_internal.h"
#include "xray_tsc.h"
#include "xray_utils.h"

namespace __xray {

SpinMutex LogMutex;

// We use elements of this type to record the entry TSC of every function ID we
// see as we're tracing a particular thread's execution.
struct alignas(16) StackEntry {
  int32_t FuncId;
  uint16_t Type;
  uint8_t CPU;
  uint8_t Padding;
  uint64_t TSC;
};

static_assert(sizeof(StackEntry) == 16, "Wrong size for StackEntry");

struct alignas(64) ThreadLocalData {
  void *InMemoryBuffer = nullptr;
  size_t BufferSize = 0;
  size_t BufferOffset = 0;
  void *ShadowStack = nullptr;
  size_t StackSize = 0;
  size_t StackEntries = 0;
  int Fd = -1;
  tid_t TID = 0;
};

static pthread_key_t PThreadKey;

static atomic_uint8_t BasicInitialized{0};

BasicLoggingOptions GlobalOptions;

thread_local volatile bool RecursionGuard = false;

static uint64_t thresholdTicks() XRAY_NEVER_INSTRUMENT {
  static uint64_t TicksPerSec = probeRequiredCPUFeatures()
                                    ? getTSCFrequency()
                                    : __xray::NanosecondsPerSecond;
  static const uint64_t ThresholdTicks =
      TicksPerSec * GlobalOptions.DurationFilterMicros / 1000000;
  return ThresholdTicks;
}

static int openLogFile() XRAY_NEVER_INSTRUMENT {
  int F = getLogFD();
  if (F == -1)
    return -1;

  // Test for required CPU features and cache the cycle frequency
  static bool TSCSupported = probeRequiredCPUFeatures();
  static uint64_t CycleFrequency =
      TSCSupported ? getTSCFrequency() : __xray::NanosecondsPerSecond;

  // Since we're here, we get to write the header. We set it up so that the
  // header will only be written once, at the start, and let the threads
  // logging do writes which just append.
  XRayFileHeader Header;
  Header.Version = 2; // Version 2 includes tail exit records.
  Header.Type = FileTypes::NAIVE_LOG;
  Header.CycleFrequency = CycleFrequency;

  // FIXME: Actually check whether we have 'constant_tsc' and 'nonstop_tsc'
  // before setting the values in the header.
  Header.ConstantTSC = 1;
  Header.NonstopTSC = 1;
  retryingWriteAll(F, reinterpret_cast<char *>(&Header),
                   reinterpret_cast<char *>(&Header) + sizeof(Header));
  return F;
}

int getGlobalFd() XRAY_NEVER_INSTRUMENT {
  static int Fd = openLogFile();
  return Fd;
}

ThreadLocalData &getThreadLocalData() XRAY_NEVER_INSTRUMENT {
  thread_local ThreadLocalData TLD;
  thread_local bool UNUSED TOnce = [] {
    if (GlobalOptions.ThreadBufferSize == 0) {
      if (Verbosity())
        Report("Not initializing TLD since ThreadBufferSize == 0.\n");
      return false;
    }
    TLD.TID = GetTid();
    pthread_setspecific(PThreadKey, &TLD);
    TLD.Fd = getGlobalFd();
    TLD.InMemoryBuffer = reinterpret_cast<XRayRecord *>(
        InternalAlloc(sizeof(XRayRecord) * GlobalOptions.ThreadBufferSize,
                      nullptr, alignof(XRayRecord)));
    TLD.BufferSize = GlobalOptions.ThreadBufferSize;
    TLD.BufferOffset = 0;
    if (GlobalOptions.MaxStackDepth == 0) {
      if (Verbosity())
        Report("Not initializing the ShadowStack since MaxStackDepth == 0.\n");
      TLD.StackSize = 0;
      TLD.StackEntries = 0;
      TLD.ShadowStack = nullptr;
      return false;
    }
    TLD.ShadowStack = reinterpret_cast<StackEntry *>(
        InternalAlloc(sizeof(StackEntry) * GlobalOptions.MaxStackDepth, nullptr,
                      alignof(StackEntry)));
    TLD.StackSize = GlobalOptions.MaxStackDepth;
    TLD.StackEntries = 0;
    if (Verbosity() >= 2) {
      static auto UNUSED Once = [] {
        auto ticks = thresholdTicks();
        Report("Ticks threshold: %d\n", ticks);
        return false;
      }();
    }
    return false;
  }();
  return TLD;
}

template <class RDTSC>
void InMemoryRawLog(int32_t FuncId, XRayEntryType Type,
                    RDTSC ReadTSC) XRAY_NEVER_INSTRUMENT {
  auto &TLD = getThreadLocalData();
  int Fd = getGlobalFd();
  if (Fd == -1)
    return;

  // Use a simple recursion guard, to handle cases where we're already logging
  // and for one reason or another, this function gets called again in the same
  // thread.
  if (RecursionGuard)
    return;
  RecursionGuard = true;
  auto ExitGuard = at_scope_exit([] { RecursionGuard = false; });

  uint8_t CPU = 0;
  uint64_t TSC = ReadTSC(CPU);

  switch (Type) {
  case XRayEntryType::ENTRY:
  case XRayEntryType::LOG_ARGS_ENTRY: {
    // Short circuit if we've reached the maximum depth of the stack.
    if (TLD.StackEntries++ >= TLD.StackSize)
      return;

    // When we encounter an entry event, we keep track of the TSC and the CPU,
    // and put it in the stack.
    StackEntry E;
    E.FuncId = FuncId;
    E.CPU = CPU;
    E.Type = Type;
    E.TSC = TSC;
    auto StackEntryPtr = static_cast<char *>(TLD.ShadowStack) +
                         (sizeof(StackEntry) * (TLD.StackEntries - 1));
    internal_memcpy(StackEntryPtr, &E, sizeof(StackEntry));
    break;
  }
  case XRayEntryType::EXIT:
  case XRayEntryType::TAIL: {
    if (TLD.StackEntries == 0)
      break;

    if (--TLD.StackEntries >= TLD.StackSize)
      return;

    // When we encounter an exit event, we check whether all the following are
    // true:
    //
    // - The Function ID is the same as the most recent entry in the stack.
    // - The CPU is the same as the most recent entry in the stack.
    // - The Delta of the TSCs is less than the threshold amount of time we're
    //   looking to record.
    //
    // If all of these conditions are true, we pop the stack and don't write a
    // record and move the record offset back.
    StackEntry StackTop;
    auto StackEntryPtr = static_cast<char *>(TLD.ShadowStack) +
                         (sizeof(StackEntry) * TLD.StackEntries);
    internal_memcpy(&StackTop, StackEntryPtr, sizeof(StackEntry));
    if (StackTop.FuncId == FuncId && StackTop.CPU == CPU &&
        StackTop.TSC < TSC) {
      auto Delta = TSC - StackTop.TSC;
      if (Delta < thresholdTicks()) {
        assert(TLD.BufferOffset > 0);
        TLD.BufferOffset -= StackTop.Type == XRayEntryType::ENTRY ? 1 : 2;
        return;
      }
    }
    break;
  }
  default:
    // Should be unreachable.
    assert(false && "Unsupported XRayEntryType encountered.");
    break;
  }

  // First determine whether the delta between the function's enter record and
  // the exit record is higher than the threshold.
  __xray::XRayRecord R;
  R.RecordType = RecordTypes::NORMAL;
  R.CPU = CPU;
  R.TSC = TSC;
  R.TId = TLD.TID;
  R.Type = Type;
  R.FuncId = FuncId;
  auto FirstEntry = reinterpret_cast<__xray::XRayRecord *>(TLD.InMemoryBuffer);
  internal_memcpy(FirstEntry + TLD.BufferOffset, &R, sizeof(R));
  if (++TLD.BufferOffset == TLD.BufferSize) {
    SpinMutexLock L(&LogMutex);
    retryingWriteAll(Fd, reinterpret_cast<char *>(FirstEntry),
                     reinterpret_cast<char *>(FirstEntry + TLD.BufferOffset));
    TLD.BufferOffset = 0;
    TLD.StackEntries = 0;
  }
}

template <class RDTSC>
void InMemoryRawLogWithArg(int32_t FuncId, XRayEntryType Type, uint64_t Arg1,
                           RDTSC ReadTSC) XRAY_NEVER_INSTRUMENT {
  auto &TLD = getThreadLocalData();
  auto FirstEntry =
      reinterpret_cast<__xray::XRayArgPayload *>(TLD.InMemoryBuffer);
  const auto &BuffLen = TLD.BufferSize;
  int Fd = getGlobalFd();
  if (Fd == -1)
    return;

  // First we check whether there's enough space to write the data consecutively
  // in the thread-local buffer. If not, we first flush the buffer before
  // attempting to write the two records that must be consecutive.
  if (TLD.BufferOffset + 2 > BuffLen) {
    SpinMutexLock L(&LogMutex);
    retryingWriteAll(Fd, reinterpret_cast<char *>(FirstEntry),
                     reinterpret_cast<char *>(FirstEntry + TLD.BufferOffset));
    TLD.BufferOffset = 0;
    TLD.StackEntries = 0;
  }

  // Then we write the "we have an argument" record.
  InMemoryRawLog(FuncId, Type, ReadTSC);

  if (RecursionGuard)
    return;
  RecursionGuard = true;
  auto ExitGuard = at_scope_exit([] { RecursionGuard = false; });

  // And from here on write the arg payload.
  __xray::XRayArgPayload R;
  R.RecordType = RecordTypes::ARG_PAYLOAD;
  R.FuncId = FuncId;
  R.TId = TLD.TID;
  R.Arg = Arg1;
  internal_memcpy(FirstEntry + TLD.BufferOffset, &R, sizeof(R));
  if (++TLD.BufferOffset == BuffLen) {
    SpinMutexLock L(&LogMutex);
    retryingWriteAll(Fd, reinterpret_cast<char *>(FirstEntry),
                     reinterpret_cast<char *>(FirstEntry + TLD.BufferOffset));
    TLD.BufferOffset = 0;
    TLD.StackEntries = 0;
  }
}

void basicLoggingHandleArg0RealTSC(int32_t FuncId,
                                   XRayEntryType Type) XRAY_NEVER_INSTRUMENT {
  InMemoryRawLog(FuncId, Type, __xray::readTSC);
}

void basicLoggingHandleArg0EmulateTSC(int32_t FuncId, XRayEntryType Type)
    XRAY_NEVER_INSTRUMENT {
  InMemoryRawLog(FuncId, Type, [](uint8_t &CPU) XRAY_NEVER_INSTRUMENT {
    timespec TS;
    int result = clock_gettime(CLOCK_REALTIME, &TS);
    if (result != 0) {
      Report("clock_gettimg(2) return %d, errno=%d.", result, int(errno));
      TS = {0, 0};
    }
    CPU = 0;
    return TS.tv_sec * __xray::NanosecondsPerSecond + TS.tv_nsec;
  });
}

void basicLoggingHandleArg1RealTSC(int32_t FuncId, XRayEntryType Type,
                                   uint64_t Arg1) XRAY_NEVER_INSTRUMENT {
  InMemoryRawLogWithArg(FuncId, Type, Arg1, __xray::readTSC);
}

void basicLoggingHandleArg1EmulateTSC(int32_t FuncId, XRayEntryType Type,
                                      uint64_t Arg1) XRAY_NEVER_INSTRUMENT {
  InMemoryRawLogWithArg(
      FuncId, Type, Arg1, [](uint8_t &CPU) XRAY_NEVER_INSTRUMENT {
        timespec TS;
        int result = clock_gettime(CLOCK_REALTIME, &TS);
        if (result != 0) {
          Report("clock_gettimg(2) return %d, errno=%d.", result, int(errno));
          TS = {0, 0};
        }
        CPU = 0;
        return TS.tv_sec * __xray::NanosecondsPerSecond + TS.tv_nsec;
      });
}

static void TLDDestructor(void *P) XRAY_NEVER_INSTRUMENT {
  ThreadLocalData &TLD = *reinterpret_cast<ThreadLocalData *>(P);
  auto ExitGuard = at_scope_exit([&TLD] {
    // Clean up dynamic resources.
    if (TLD.InMemoryBuffer)
      InternalFree(TLD.InMemoryBuffer);
    if (TLD.ShadowStack)
      InternalFree(TLD.ShadowStack);
    if (Verbosity())
      Report("Cleaned up log for TID: %d\n", TLD.TID);
  });

  if (TLD.Fd == -1 || TLD.BufferOffset == 0) {
    if (Verbosity())
      Report("Skipping buffer for TID: %d; Fd = %d; Offset = %llu\n", TLD.TID,
             TLD.Fd, TLD.BufferOffset);
    return;
  }

  {
    SpinMutexLock L(&LogMutex);
    retryingWriteAll(TLD.Fd, reinterpret_cast<char *>(TLD.InMemoryBuffer),
                     reinterpret_cast<char *>(TLD.InMemoryBuffer) +
                         (sizeof(__xray::XRayRecord) * TLD.BufferOffset));
  }

  // Because this thread's exit could be the last one trying to write to
  // the file and that we're not able to close out the file properly, we
  // sync instead and hope that the pending writes are flushed as the
  // thread exits.
  fsync(TLD.Fd);
}

XRayLogInitStatus basicLoggingInit(size_t BufferSize, size_t BufferMax,
                                   void *Options,
                                   size_t OptionsSize) XRAY_NEVER_INSTRUMENT {
  uint8_t Expected = 0;
  if (!atomic_compare_exchange_strong(
          &BasicInitialized, &Expected, 1, memory_order_acq_rel)) {
    if (Verbosity())
      Report("Basic logging already initialized.\n");
    return XRayLogInitStatus::XRAY_LOG_INITIALIZED;
  }

  static bool UNUSED Once = [] {
    pthread_key_create(&PThreadKey, TLDDestructor);
    return false;
  }();

  if (BufferSize == 0 && BufferMax == 0 && Options != nullptr) {
    FlagParser P;
    BasicFlags F;
    F.setDefaults();
    registerXRayBasicFlags(&P, &F);
    P.ParseString(useCompilerDefinedBasicFlags());
    auto *EnvOpts = GetEnv("XRAY_BASIC_OPTIONS");
    if (EnvOpts == nullptr)
      EnvOpts = "";

    P.ParseString(EnvOpts);

    // If XRAY_BASIC_OPTIONS was not defined, then we use the deprecated options
    // set through XRAY_OPTIONS instead.
    if (internal_strlen(EnvOpts) == 0) {
      F.func_duration_threshold_us =
          flags()->xray_naive_log_func_duration_threshold_us;
      F.max_stack_depth = flags()->xray_naive_log_max_stack_depth;
      F.thread_buffer_size = flags()->xray_naive_log_thread_buffer_size;
    }

    P.ParseString(static_cast<const char *>(Options));
    GlobalOptions.ThreadBufferSize = F.thread_buffer_size;
    GlobalOptions.DurationFilterMicros = F.func_duration_threshold_us;
    GlobalOptions.MaxStackDepth = F.max_stack_depth;
  } else if (OptionsSize != sizeof(BasicLoggingOptions)) {
    Report("Invalid options size, potential ABI mismatch; expected %d got %d",
           sizeof(BasicLoggingOptions), OptionsSize);
    return XRayLogInitStatus::XRAY_LOG_UNINITIALIZED;
  } else {
    if (Verbosity())
      Report("XRay Basic: struct-based init is deprecated, please use "
             "string-based configuration instead.\n");
    GlobalOptions = *reinterpret_cast<BasicLoggingOptions *>(Options);
  }

  static auto UseRealTSC = probeRequiredCPUFeatures();
  if (!UseRealTSC && Verbosity())
    Report("WARNING: Required CPU features missing for XRay instrumentation, "
           "using emulation instead.\n");

  __xray_set_handler_arg1(UseRealTSC ? basicLoggingHandleArg1RealTSC
                                     : basicLoggingHandleArg1EmulateTSC);
  __xray_set_handler(UseRealTSC ? basicLoggingHandleArg0RealTSC
                                : basicLoggingHandleArg0EmulateTSC);
  __xray_remove_customevent_handler();
  __xray_remove_typedevent_handler();

  return XRayLogInitStatus::XRAY_LOG_INITIALIZED;
}

XRayLogInitStatus basicLoggingFinalize() XRAY_NEVER_INSTRUMENT {
  uint8_t Expected = 0;
  if (!atomic_compare_exchange_strong(
          &BasicInitialized, &Expected, 0, memory_order_acq_rel) &&
      Verbosity())
    Report("Basic logging already finalized.\n");

  // Nothing really to do aside from marking state of the global to be
  // uninitialized.

  return XRayLogInitStatus::XRAY_LOG_FINALIZED;
}

XRayLogFlushStatus basicLoggingFlush() XRAY_NEVER_INSTRUMENT {
  // This really does nothing, since flushing the logs happen at the end of a
  // thread's lifetime, or when the buffers are full.
  return XRayLogFlushStatus::XRAY_LOG_FLUSHED;
}

// This is a handler that, effectively, does nothing.
void basicLoggingHandleArg0Empty(int32_t, XRayEntryType) XRAY_NEVER_INSTRUMENT {
}

bool basicLogDynamicInitializer() XRAY_NEVER_INSTRUMENT {
  XRayLogImpl Impl{
      basicLoggingInit,
      basicLoggingFinalize,
      basicLoggingHandleArg0Empty,
      basicLoggingFlush,
  };
  auto RegistrationResult = __xray_log_register_mode("xray-basic", Impl);
  if (RegistrationResult != XRayLogRegisterStatus::XRAY_REGISTRATION_OK &&
      Verbosity())
    Report("Cannot register XRay Basic Mode to 'xray-basic'; error = %d\n",
           RegistrationResult);
  if (flags()->xray_naive_log ||
      !internal_strcmp(flags()->xray_mode, "xray-basic")) {
    auto SelectResult = __xray_log_select_mode("xray-basic");
    if (SelectResult != XRayLogRegisterStatus::XRAY_REGISTRATION_OK) {
      if (Verbosity())
        Report("Failed selecting XRay Basic Mode; error = %d\n", SelectResult);
      return false;
    }

    // We initialize the implementation using the data we get from the
    // XRAY_BASIC_OPTIONS environment variable, at this point of the
    // implementation.
    auto *Env = GetEnv("XRAY_BASIC_OPTIONS");
    auto InitResult =
        __xray_log_init_mode("xray-basic", Env == nullptr ? "" : Env);
    if (InitResult != XRayLogInitStatus::XRAY_LOG_INITIALIZED) {
      if (Verbosity())
        Report("Failed initializing XRay Basic Mode; error = %d\n", InitResult);
      return false;
    }
    static auto UNUSED Once = [] {
      static auto UNUSED &TLD = getThreadLocalData();
      Atexit(+[] { TLDDestructor(&TLD); });
      return false;
    }();
  }
  return true;
}

} // namespace __xray

static auto UNUSED Unused = __xray::basicLogDynamicInitializer();
