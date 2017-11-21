//===-- xray_inmemory_log.cc ------------------------------------*- C++ -*-===//
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
#include "xray_defs.h"
#include "xray_flags.h"
#include "xray_inmemory_log.h"
#include "xray_interface_internal.h"
#include "xray_tsc.h"
#include "xray_utils.h"

namespace __xray {

__sanitizer::SpinMutex LogMutex;

// We use elements of this type to record the entry TSC of every function ID we
// see as we're tracing a particular thread's execution.
struct StackEntry {
  int32_t FuncId;
  uint64_t TSC;
};

struct alignas(64) ThreadLocalData {
  XRayRecord *InMemoryBuffer = nullptr;
  size_t BufferSize = 0;
  size_t BufferOffset = 0;
  StackEntry *ShadowStack = nullptr;
  size_t StackSize = 0;
  size_t StackEntries = 0;
  int Fd = -1;
};

static pthread_key_t PThreadKey;

static __sanitizer::atomic_uint8_t BasicInitialized{0};

BasicLoggingOptions GlobalOptions;

thread_local volatile bool RecusionGuard = false;

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

pid_t getTId() XRAY_NEVER_INSTRUMENT {
  thread_local pid_t TId = syscall(SYS_gettid);
  return TId;
}

int getGlobalFd() XRAY_NEVER_INSTRUMENT {
  static int Fd = openLogFile();
  return Fd;
}

ThreadLocalData &getThreadLocalData() XRAY_NEVER_INSTRUMENT {
  thread_local ThreadLocalData TLD;
  thread_local bool UNUSED TOnce = [] {
    if (GlobalOptions.ThreadBufferSize == 0)
      return false;
    pthread_setspecific(PThreadKey, &TLD);
    TLD.Fd = getGlobalFd();
    TLD.InMemoryBuffer = reinterpret_cast<XRayRecord *>(
        InternalAlloc(sizeof(XRayRecord) * GlobalOptions.ThreadBufferSize,
                      nullptr, alignof(XRayRecord)));
    TLD.BufferSize = GlobalOptions.ThreadBufferSize;
    TLD.BufferOffset = 0;
    if (GlobalOptions.MaxStackDepth == 0)
      return false;
    TLD.ShadowStack = reinterpret_cast<StackEntry *>(
        InternalAlloc(sizeof(StackEntry) * GlobalOptions.MaxStackDepth, nullptr,
                      alignof(StackEntry)));
    TLD.StackSize = GlobalOptions.MaxStackDepth;
    TLD.StackEntries = 0;
    return false;
  }();
  return TLD;
}

template <class RDTSC>
void InMemoryRawLog(int32_t FuncId, XRayEntryType Type,
                    RDTSC ReadTSC) XRAY_NEVER_INSTRUMENT {
  auto &TLD = getThreadLocalData();
  auto &InMemoryBuffer = TLD.InMemoryBuffer;
  auto &Offset = TLD.BufferOffset;
  int Fd = getGlobalFd();
  if (Fd == -1)
    return;

  // Use a simple recursion guard, to handle cases where we're already logging
  // and for one reason or another, this function gets called again in the same
  // thread.
  if (RecusionGuard)
    return;
  RecusionGuard = true;

  // First we get the useful data, and stuff it into the already aligned buffer
  // through a pointer offset.
  auto &R = reinterpret_cast<__xray::XRayRecord *>(InMemoryBuffer)[Offset];
  R.RecordType = RecordTypes::NORMAL;
  R.TSC = ReadTSC(R.CPU);
  R.TId = getTId();
  R.Type = Type;
  R.FuncId = FuncId;
  if (++Offset == TLD.BufferSize) {
    __sanitizer::SpinMutexLock L(&LogMutex);
    auto RecordBuffer = reinterpret_cast<__xray::XRayRecord *>(InMemoryBuffer);
    retryingWriteAll(Fd, reinterpret_cast<char *>(RecordBuffer),
                     reinterpret_cast<char *>(RecordBuffer + Offset));
    Offset = 0;
  }

  RecusionGuard = false;
}

template <class RDTSC>
void InMemoryRawLogWithArg(int32_t FuncId, XRayEntryType Type, uint64_t Arg1,
                           RDTSC ReadTSC) XRAY_NEVER_INSTRUMENT {
  auto &TLD = getThreadLocalData();
  auto &InMemoryBuffer = TLD.InMemoryBuffer;
  auto &Offset = TLD.BufferOffset;
  const auto &BuffLen = TLD.BufferSize;
  int Fd = getGlobalFd();
  if (Fd == -1)
    return;

  // First we check whether there's enough space to write the data consecutively
  // in the thread-local buffer. If not, we first flush the buffer before
  // attempting to write the two records that must be consecutive.
  if (Offset + 2 > BuffLen) {
    __sanitizer::SpinMutexLock L(&LogMutex);
    auto RecordBuffer = reinterpret_cast<__xray::XRayRecord *>(InMemoryBuffer);
    retryingWriteAll(Fd, reinterpret_cast<char *>(RecordBuffer),
                     reinterpret_cast<char *>(RecordBuffer + Offset));
    Offset = 0;
  }

  // Then we write the "we have an argument" record.
  InMemoryRawLog(FuncId, Type, ReadTSC);

  if (RecusionGuard)
    return;

  RecusionGuard = true;

  // And from here on write the arg payload.
  __xray::XRayArgPayload R;
  R.RecordType = RecordTypes::ARG_PAYLOAD;
  R.FuncId = FuncId;
  R.TId = getTId();
  R.Arg = Arg1;
  auto EntryPtr =
      &reinterpret_cast<__xray::XRayArgPayload *>(&InMemoryBuffer)[Offset];
  std::memcpy(EntryPtr, &R, sizeof(R));
  if (++Offset == BuffLen) {
    __sanitizer::SpinMutexLock L(&LogMutex);
    auto RecordBuffer = reinterpret_cast<__xray::XRayRecord *>(InMemoryBuffer);
    retryingWriteAll(Fd, reinterpret_cast<char *>(RecordBuffer),
                     reinterpret_cast<char *>(RecordBuffer + Offset));
    Offset = 0;
  }

  RecusionGuard = false;
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

XRayLogInitStatus basicLoggingInit(size_t BufferSize, size_t BufferMax,
                                   void *Options,
                                   size_t OptionsSize) XRAY_NEVER_INSTRUMENT {
  static bool UNUSED Once = [] {
    pthread_key_create(&PThreadKey, +[](void *P) {
      ThreadLocalData &TLD = *reinterpret_cast<ThreadLocalData *>(P);
      if (TLD.Fd == -1 || TLD.BufferOffset == 0)
        return;

      {
        __sanitizer::SpinMutexLock L(&LogMutex);
        retryingWriteAll(
            TLD.Fd, reinterpret_cast<char *>(TLD.InMemoryBuffer),
            reinterpret_cast<char *>(TLD.InMemoryBuffer + TLD.BufferOffset));
      }

      // Because this thread's exit could be the last one trying to write to
      // the file and that we're not able to close out the file properly, we
      // sync instead and hope that the pending writes are flushed as the
      // thread exits.
      fsync(TLD.Fd);

      // Clean up dynamic resources.
      if (TLD.InMemoryBuffer)
        InternalFree(TLD.InMemoryBuffer);
      if (TLD.ShadowStack)
        InternalFree(TLD.ShadowStack);
    });
    return false;
  }();

  uint8_t Expected = 0;
  if (!__sanitizer::atomic_compare_exchange_strong(
          &BasicInitialized, &Expected, 1, __sanitizer::memory_order_acq_rel)) {
    if (__sanitizer::Verbosity())
      Report("Basic logging already initialized.\n");
    return XRayLogInitStatus::XRAY_LOG_INITIALIZED;
  }

  if (OptionsSize != sizeof(BasicLoggingOptions)) {
    Report("Invalid options size, potential ABI mismatch; expected %d got %d",
           sizeof(BasicLoggingOptions), OptionsSize);
    return XRayLogInitStatus::XRAY_LOG_UNINITIALIZED;
  }

  static auto UseRealTSC = probeRequiredCPUFeatures();
  if (!UseRealTSC && __sanitizer::Verbosity())
    Report("WARNING: Required CPU features missing for XRay instrumentation, "
           "using emulation instead.\n");

  GlobalOptions = *reinterpret_cast<BasicLoggingOptions *>(Options);
  __xray_set_handler_arg1(UseRealTSC ? basicLoggingHandleArg1RealTSC
                                     : basicLoggingHandleArg1EmulateTSC);
  __xray_set_handler(UseRealTSC ? basicLoggingHandleArg0RealTSC
                                : basicLoggingHandleArg0EmulateTSC);
  __xray_remove_customevent_handler();
  return XRayLogInitStatus::XRAY_LOG_INITIALIZED;
}

XRayLogInitStatus basicLoggingFinalize() XRAY_NEVER_INSTRUMENT {
  uint8_t Expected = 0;
  if (!__sanitizer::atomic_compare_exchange_strong(
          &BasicInitialized, &Expected, 0, __sanitizer::memory_order_acq_rel) &&
      __sanitizer::Verbosity())
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
  if (flags()->xray_naive_log) {
    XRayLogImpl Impl{
        basicLoggingInit,
        basicLoggingFinalize,
        basicLoggingHandleArg0Empty,
        basicLoggingFlush,
    };
    __xray_set_log_impl(Impl);
    BasicLoggingOptions Options;
    Options.DurationFilterMicros =
        flags()->xray_naive_log_func_duration_threshold_us;
    Options.MaxStackDepth = flags()->xray_naive_log_max_stack_depth;
    Options.ThreadBufferSize = flags()->xray_naive_log_thread_buffer_size;
    __xray_log_init(flags()->xray_naive_log_thread_buffer_size, 0, &Options,
                    sizeof(BasicLoggingOptions));
  }
  return true;
}

} // namespace __xray

static auto UNUSED Unused = __xray::basicLogDynamicInitializer();
