//===-- xray_fdr_logging.cc ------------------------------------*- C++ -*-===//
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
// Here we implement the Flight Data Recorder mode for XRay, where we use
// compact structures to store records in memory as well as when writing out the
// data to files.
//
//===----------------------------------------------------------------------===//
#include "xray_fdr_logging.h"
#include <algorithm>
#include <bitset>
#include <cerrno>
#include <cstring>
#include <sys/syscall.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <unordered_map>

#include "sanitizer_common/sanitizer_atomic.h"
#include "sanitizer_common/sanitizer_common.h"
#include "xray/xray_interface.h"
#include "xray/xray_records.h"
#include "xray_buffer_queue.h"
#include "xray_defs.h"
#include "xray_fdr_logging_impl.h"
#include "xray_flags.h"
#include "xray_tsc.h"
#include "xray_utils.h"

namespace __xray {

// Global BufferQueue.
// NOTE: This is a pointer to avoid having to do atomic operations at
// initialization time. This is OK to leak as there will only be one bufferqueue
// for the runtime, initialized once through the fdrInit(...) sequence.
std::shared_ptr<BufferQueue>* BQ = nullptr;

__sanitizer::atomic_sint32_t LogFlushStatus = {
    XRayLogFlushStatus::XRAY_LOG_NOT_FLUSHING};

FDRLoggingOptions FDROptions;

__sanitizer::SpinMutex FDROptionsMutex;

// Must finalize before flushing.
XRayLogFlushStatus fdrLoggingFlush() XRAY_NEVER_INSTRUMENT {
  if (__sanitizer::atomic_load(&LoggingStatus,
                               __sanitizer::memory_order_acquire) !=
      XRayLogInitStatus::XRAY_LOG_FINALIZED)
    return XRayLogFlushStatus::XRAY_LOG_NOT_FLUSHING;

  s32 Result = XRayLogFlushStatus::XRAY_LOG_NOT_FLUSHING;
  if (!__sanitizer::atomic_compare_exchange_strong(
          &LogFlushStatus, &Result, XRayLogFlushStatus::XRAY_LOG_FLUSHING,
          __sanitizer::memory_order_release))
    return static_cast<XRayLogFlushStatus>(Result);

  // Make a copy of the BufferQueue pointer to prevent other threads that may be
  // resetting it from blowing away the queue prematurely while we're dealing
  // with it.
  auto LocalBQ = *BQ;

  // We write out the file in the following format:
  //
  //   1) We write down the XRay file header with version 1, type FDR_LOG.
  //   2) Then we use the 'apply' member of the BufferQueue that's live, to
  //      ensure that at this point in time we write down the buffers that have
  //      been released (and marked "used") -- we dump the full buffer for now
  //      (fixed-sized) and let the tools reading the buffers deal with the data
  //      afterwards.
  //
  int Fd = -1;
  {
    __sanitizer::SpinMutexLock Guard(&FDROptionsMutex);
    Fd = FDROptions.Fd;
  }
  if (Fd == -1)
    Fd = getLogFD();
  if (Fd == -1) {
    auto Result = XRayLogFlushStatus::XRAY_LOG_NOT_FLUSHING;
    __sanitizer::atomic_store(&LogFlushStatus, Result,
                              __sanitizer::memory_order_release);
    return Result;
  }

  // Test for required CPU features and cache the cycle frequency
  static bool TSCSupported = probeRequiredCPUFeatures();
  static uint64_t CycleFrequency =
      TSCSupported ? getTSCFrequency() : __xray::NanosecondsPerSecond;

  XRayFileHeader Header;
  Header.Version = 1;
  Header.Type = FileTypes::FDR_LOG;
  Header.CycleFrequency = CycleFrequency;
  // FIXME: Actually check whether we have 'constant_tsc' and 'nonstop_tsc'
  // before setting the values in the header.
  Header.ConstantTSC = 1;
  Header.NonstopTSC = 1;
  Header.FdrData = FdrAdditionalHeaderData{LocalBQ->ConfiguredBufferSize()};
  retryingWriteAll(Fd, reinterpret_cast<char *>(&Header),
                   reinterpret_cast<char *>(&Header) + sizeof(Header));

  LocalBQ->apply([&](const BufferQueue::Buffer &B) {
    uint64_t BufferSize = B.Size;
    if (BufferSize > 0) {
      retryingWriteAll(Fd, reinterpret_cast<char *>(B.Buffer),
                       reinterpret_cast<char *>(B.Buffer) + B.Size);
    }
  });
  __sanitizer::atomic_store(&LogFlushStatus,
                            XRayLogFlushStatus::XRAY_LOG_FLUSHED,
                            __sanitizer::memory_order_release);
  return XRayLogFlushStatus::XRAY_LOG_FLUSHED;
}

XRayLogInitStatus fdrLoggingFinalize() XRAY_NEVER_INSTRUMENT {
  s32 CurrentStatus = XRayLogInitStatus::XRAY_LOG_INITIALIZED;
  if (!__sanitizer::atomic_compare_exchange_strong(
          &LoggingStatus, &CurrentStatus,
          XRayLogInitStatus::XRAY_LOG_FINALIZING,
          __sanitizer::memory_order_release))
    return static_cast<XRayLogInitStatus>(CurrentStatus);

  // Do special things to make the log finalize itself, and not allow any more
  // operations to be performed until re-initialized.
  (*BQ)->finalize();

  __sanitizer::atomic_store(&LoggingStatus,
                            XRayLogInitStatus::XRAY_LOG_FINALIZED,
                            __sanitizer::memory_order_release);
  return XRayLogInitStatus::XRAY_LOG_FINALIZED;
}

XRayLogInitStatus fdrLoggingReset() XRAY_NEVER_INSTRUMENT {
  s32 CurrentStatus = XRayLogInitStatus::XRAY_LOG_FINALIZED;
  if (__sanitizer::atomic_compare_exchange_strong(
          &LoggingStatus, &CurrentStatus,
          XRayLogInitStatus::XRAY_LOG_INITIALIZED,
          __sanitizer::memory_order_release))
    return static_cast<XRayLogInitStatus>(CurrentStatus);

  // Release the in-memory buffer queue.
  BQ->reset();

  // Spin until the flushing status is flushed.
  s32 CurrentFlushingStatus = XRayLogFlushStatus::XRAY_LOG_FLUSHED;
  while (__sanitizer::atomic_compare_exchange_weak(
      &LogFlushStatus, &CurrentFlushingStatus,
      XRayLogFlushStatus::XRAY_LOG_NOT_FLUSHING,
      __sanitizer::memory_order_release)) {
    if (CurrentFlushingStatus == XRayLogFlushStatus::XRAY_LOG_NOT_FLUSHING)
      break;
    CurrentFlushingStatus = XRayLogFlushStatus::XRAY_LOG_FLUSHED;
  }

  // At this point, we know that the status is flushed, and that we can assume
  return XRayLogInitStatus::XRAY_LOG_UNINITIALIZED;
}

static std::tuple<uint64_t, unsigned char>
getTimestamp() XRAY_NEVER_INSTRUMENT {
  // We want to get the TSC as early as possible, so that we can check whether
  // we've seen this CPU before. We also do it before we load anything else, to
  // allow for forward progress with the scheduling.
  unsigned char CPU;
  uint64_t TSC;

  // Test once for required CPU features
  static bool TSCSupported = probeRequiredCPUFeatures();

  if (TSCSupported) {
    TSC = __xray::readTSC(CPU);
  } else {
    // FIXME: This code needs refactoring as it appears in multiple locations
    timespec TS;
    int result = clock_gettime(CLOCK_REALTIME, &TS);
    if (result != 0) {
      Report("clock_gettime(2) return %d, errno=%d", result, int(errno));
      TS = {0, 0};
    }
    CPU = 0;
    TSC = TS.tv_sec * __xray::NanosecondsPerSecond + TS.tv_nsec;
  }
  return std::make_tuple(TSC, CPU);
}

void fdrLoggingHandleArg0(int32_t FuncId,
                          XRayEntryType Entry) XRAY_NEVER_INSTRUMENT {
  auto TSC_CPU = getTimestamp();
  __xray_fdr_internal::processFunctionHook(FuncId, Entry, std::get<0>(TSC_CPU),
                                           std::get<1>(TSC_CPU), clock_gettime,
                                           LoggingStatus, *BQ);
}

void fdrLoggingHandleCustomEvent(void *Event,
                                 std::size_t EventSize) XRAY_NEVER_INSTRUMENT {
  using namespace __xray_fdr_internal;
  auto TSC_CPU = getTimestamp();
  auto &TSC = std::get<0>(TSC_CPU);
  auto &CPU = std::get<1>(TSC_CPU);
  thread_local bool Running = false;
  RecursionGuard Guard{Running};
  if (!Guard) {
    assert(Running && "RecursionGuard is buggy!");
    return;
  }
  if (EventSize > std::numeric_limits<int32_t>::max()) {
    using Empty = struct {};
    static Empty Once = [&] {
      Report("Event size too large = %zu ; > max = %d\n", EventSize,
             std::numeric_limits<int32_t>::max());
      return Empty();
    }();
    (void)Once;
  }
  int32_t ReducedEventSize = static_cast<int32_t>(EventSize);
  auto &TLD = getThreadLocalData();
  if (!isLogInitializedAndReady(TLD.LocalBQ, TSC, CPU, clock_gettime))
    return;

  // Here we need to prepare the log to handle:
  //   - The metadata record we're going to write. (16 bytes)
  //   - The additional data we're going to write. Currently, that's the size of
  //   the event we're going to dump into the log as free-form bytes.
  if (!prepareBuffer(clock_gettime, MetadataRecSize + EventSize)) {
    TLD.LocalBQ = nullptr;
    return;
  }

  // Write the custom event metadata record, which consists of the following
  // information:
  //   - 8 bytes (64-bits) for the full TSC when the event started.
  //   - 4 bytes (32-bits) for the length of the data.
  MetadataRecord CustomEvent;
  CustomEvent.Type = uint8_t(RecordType::Metadata);
  CustomEvent.RecordKind =
      uint8_t(MetadataRecord::RecordKinds::CustomEventMarker);
  constexpr auto TSCSize = sizeof(std::get<0>(TSC_CPU));
  std::memcpy(&CustomEvent.Data, &ReducedEventSize, sizeof(int32_t));
  std::memcpy(&CustomEvent.Data[sizeof(int32_t)], &TSC, TSCSize);
  std::memcpy(TLD.RecordPtr, &CustomEvent, sizeof(CustomEvent));
  TLD.RecordPtr += sizeof(CustomEvent);
  std::memcpy(TLD.RecordPtr, Event, ReducedEventSize);
  endBufferIfFull();
}

XRayLogInitStatus fdrLoggingInit(std::size_t BufferSize, std::size_t BufferMax,
                                 void *Options,
                                 size_t OptionsSize) XRAY_NEVER_INSTRUMENT {
  if (OptionsSize != sizeof(FDRLoggingOptions))
    return static_cast<XRayLogInitStatus>(__sanitizer::atomic_load(
        &LoggingStatus, __sanitizer::memory_order_acquire));
  s32 CurrentStatus = XRayLogInitStatus::XRAY_LOG_UNINITIALIZED;
  if (!__sanitizer::atomic_compare_exchange_strong(
          &LoggingStatus, &CurrentStatus,
          XRayLogInitStatus::XRAY_LOG_INITIALIZING,
          __sanitizer::memory_order_release))
    return static_cast<XRayLogInitStatus>(CurrentStatus);

  {
    __sanitizer::SpinMutexLock Guard(&FDROptionsMutex);
    memcpy(&FDROptions, Options, OptionsSize);
  }

  bool Success = false;
  if (BQ == nullptr)
    BQ = new std::shared_ptr<BufferQueue>();

  *BQ = std::make_shared<BufferQueue>(BufferSize, BufferMax, Success);
  if (!Success) {
    Report("BufferQueue init failed.\n");
    return XRayLogInitStatus::XRAY_LOG_UNINITIALIZED;
  }

  // Install the actual handleArg0 handler after initialising the buffers.
  __xray_set_handler(fdrLoggingHandleArg0);
  __xray_set_customevent_handler(fdrLoggingHandleCustomEvent);

  __sanitizer::atomic_store(&LoggingStatus,
                            XRayLogInitStatus::XRAY_LOG_INITIALIZED,
                            __sanitizer::memory_order_release);
  Report("XRay FDR init successful.\n");
  return XRayLogInitStatus::XRAY_LOG_INITIALIZED;
}

} // namespace __xray

static auto UNUSED Unused = [] {
  using namespace __xray;
  if (flags()->xray_fdr_log) {
    XRayLogImpl Impl{
        fdrLoggingInit, fdrLoggingFinalize, fdrLoggingHandleArg0,
        fdrLoggingFlush,
    };
    __xray_set_log_impl(Impl);
  }
  return true;
}();
