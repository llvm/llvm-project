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
#include <errno.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

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
BufferQueue *BQ = nullptr;

__sanitizer::atomic_sint32_t LogFlushStatus = {
    XRayLogFlushStatus::XRAY_LOG_NOT_FLUSHING};

FDRLoggingOptions FDROptions;

__sanitizer::SpinMutex FDROptionsMutex;

// Must finalize before flushing.
XRayLogFlushStatus fdrLoggingFlush() XRAY_NEVER_INSTRUMENT {
  if (__sanitizer::atomic_load(&LoggingStatus,
                               __sanitizer::memory_order_acquire) !=
      XRayLogInitStatus::XRAY_LOG_FINALIZED) {
    if (__sanitizer::Verbosity())
      Report("Not flushing log, implementation is not finalized.\n");
    return XRayLogFlushStatus::XRAY_LOG_NOT_FLUSHING;
  }

  s32 Result = XRayLogFlushStatus::XRAY_LOG_NOT_FLUSHING;
  if (!__sanitizer::atomic_compare_exchange_strong(
          &LogFlushStatus, &Result, XRayLogFlushStatus::XRAY_LOG_FLUSHING,
          __sanitizer::memory_order_release)) {

    if (__sanitizer::Verbosity())
      Report("Not flushing log, implementation is still finalizing.\n");
    return static_cast<XRayLogFlushStatus>(Result);
  }

  if (BQ == nullptr) {
    if (__sanitizer::Verbosity())
      Report("Cannot flush when global buffer queue is null.\n");
    return XRayLogFlushStatus::XRAY_LOG_NOT_FLUSHING;
  }

  // We wait a number of microseconds to allow threads to see that we've
  // finalised before attempting to flush the log.
  struct timespec TS;
  TS.tv_sec = flags()->xray_fdr_log_grace_period_us / 1000000;
  TS.tv_nsec = (flags()->xray_fdr_log_grace_period_us % 1000000) * 1000;
  struct timespec Rem;
  while (clock_nanosleep(CLOCK_REALTIME, 0, &TS, &Rem) &&
         (Rem.tv_sec != 0 || Rem.tv_nsec != 0))
    TS = Rem;

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

  // Version 2 of the log writes the extents of the buffer, instead of relying
  // on an end-of-buffer record.
  Header.Version = 2;
  Header.Type = FileTypes::FDR_LOG;
  Header.CycleFrequency = CycleFrequency;

  // FIXME: Actually check whether we have 'constant_tsc' and 'nonstop_tsc'
  // before setting the values in the header.
  Header.ConstantTSC = 1;
  Header.NonstopTSC = 1;
  Header.FdrData = FdrAdditionalHeaderData{BQ->ConfiguredBufferSize()};
  retryingWriteAll(Fd, reinterpret_cast<char *>(&Header),
                   reinterpret_cast<char *>(&Header) + sizeof(Header));

  BQ->apply([&](const BufferQueue::Buffer &B) {
    // Starting at version 2 of the FDR logging implementation, we only write
    // the records identified by the extents of the buffer. We use the Extents
    // from the Buffer and write that out as the first record in the buffer.
    // We still use a Metadata record, but fill in the extents instead for the
    // data.
    MetadataRecord ExtentsRecord;
    auto BufferExtents = __sanitizer::atomic_load(
        &B.Extents->Size, __sanitizer::memory_order_acquire);
    assert(BufferExtents <= B.Size);
    ExtentsRecord.Type = uint8_t(RecordType::Metadata);
    ExtentsRecord.RecordKind =
        uint8_t(MetadataRecord::RecordKinds::BufferExtents);
    std::memcpy(ExtentsRecord.Data, &BufferExtents, sizeof(BufferExtents));
    if (BufferExtents > 0) {
      retryingWriteAll(Fd, reinterpret_cast<char *>(&ExtentsRecord),
                       reinterpret_cast<char *>(&ExtentsRecord) +
                           sizeof(MetadataRecord));
      retryingWriteAll(Fd, reinterpret_cast<char *>(B.Buffer),
                       reinterpret_cast<char *>(B.Buffer) + BufferExtents);
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
          __sanitizer::memory_order_release)) {
    if (__sanitizer::Verbosity())
      Report("Cannot finalize log, implementation not initialized.\n");
    return static_cast<XRayLogInitStatus>(CurrentStatus);
  }

  // Do special things to make the log finalize itself, and not allow any more
  // operations to be performed until re-initialized.
  BQ->finalize();

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
  delete BQ;
  BQ = nullptr;

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

struct TSCAndCPU {
  uint64_t TSC = 0;
  unsigned char CPU = 0;
};

static TSCAndCPU getTimestamp() XRAY_NEVER_INSTRUMENT {
  // We want to get the TSC as early as possible, so that we can check whether
  // we've seen this CPU before. We also do it before we load anything else, to
  // allow for forward progress with the scheduling.
  TSCAndCPU Result;

  // Test once for required CPU features
  static bool TSCSupported = probeRequiredCPUFeatures();

  if (TSCSupported) {
    Result.TSC = __xray::readTSC(Result.CPU);
  } else {
    // FIXME: This code needs refactoring as it appears in multiple locations
    timespec TS;
    int result = clock_gettime(CLOCK_REALTIME, &TS);
    if (result != 0) {
      Report("clock_gettime(2) return %d, errno=%d", result, int(errno));
      TS = {0, 0};
    }
    Result.CPU = 0;
    Result.TSC = TS.tv_sec * __xray::NanosecondsPerSecond + TS.tv_nsec;
  }
  return Result;
}

void fdrLoggingHandleArg0(int32_t FuncId,
                          XRayEntryType Entry) XRAY_NEVER_INSTRUMENT {
  auto TC = getTimestamp();
  __xray_fdr_internal::processFunctionHook(FuncId, Entry, TC.TSC, TC.CPU, 0,
                                           clock_gettime, BQ);
}

void fdrLoggingHandleArg1(int32_t FuncId, XRayEntryType Entry,
                          uint64_t Arg) XRAY_NEVER_INSTRUMENT {
  auto TC = getTimestamp();
  __xray_fdr_internal::processFunctionHook(FuncId, Entry, TC.TSC, TC.CPU, Arg,
                                           clock_gettime, BQ);
}

void fdrLoggingHandleCustomEvent(void *Event,
                                 std::size_t EventSize) XRAY_NEVER_INSTRUMENT {
  using namespace __xray_fdr_internal;
  auto TC = getTimestamp();
  auto &TSC = TC.TSC;
  auto &CPU = TC.CPU;
  RecursionGuard Guard{Running};
  if (!Guard)
    return;
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
  if (!isLogInitializedAndReady(TLD.BQ, TSC, CPU, clock_gettime))
    return;

  // Here we need to prepare the log to handle:
  //   - The metadata record we're going to write. (16 bytes)
  //   - The additional data we're going to write. Currently, that's the size of
  //   the event we're going to dump into the log as free-form bytes.
  if (!prepareBuffer(TSC, CPU, clock_gettime, MetadataRecSize + EventSize)) {
    TLD.BQ = nullptr;
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
  constexpr auto TSCSize = sizeof(TC.TSC);
  std::memcpy(&CustomEvent.Data, &ReducedEventSize, sizeof(int32_t));
  std::memcpy(&CustomEvent.Data[sizeof(int32_t)], &TSC, TSCSize);
  std::memcpy(TLD.RecordPtr, &CustomEvent, sizeof(CustomEvent));
  TLD.RecordPtr += sizeof(CustomEvent);
  std::memcpy(TLD.RecordPtr, Event, ReducedEventSize);
  incrementExtents(MetadataRecSize + EventSize);
  endBufferIfFull();
}

XRayLogInitStatus fdrLoggingInit(std::size_t BufferSize, std::size_t BufferMax,
                                 void *Options,
                                 size_t OptionsSize) XRAY_NEVER_INSTRUMENT {
  if (OptionsSize != sizeof(FDRLoggingOptions)) {
    if (__sanitizer::Verbosity())
      Report("Cannot initialize FDR logging; wrong size for options: %d\n",
             OptionsSize);
    return static_cast<XRayLogInitStatus>(__sanitizer::atomic_load(
        &LoggingStatus, __sanitizer::memory_order_acquire));
  }
  s32 CurrentStatus = XRayLogInitStatus::XRAY_LOG_UNINITIALIZED;
  if (!__sanitizer::atomic_compare_exchange_strong(
          &LoggingStatus, &CurrentStatus,
          XRayLogInitStatus::XRAY_LOG_INITIALIZING,
          __sanitizer::memory_order_release)) {
    if (__sanitizer::Verbosity())
      Report("Cannot initialize already initialized implementation.\n");
    return static_cast<XRayLogInitStatus>(CurrentStatus);
  }

  {
    __sanitizer::SpinMutexLock Guard(&FDROptionsMutex);
    memcpy(&FDROptions, Options, OptionsSize);
  }

  bool Success = false;

  if (BQ != nullptr) {
    delete BQ;
    BQ = nullptr;
  }

  if (BQ == nullptr)
    BQ = new BufferQueue(BufferSize, BufferMax, Success);

  if (!Success) {
    Report("BufferQueue init failed.\n");
    if (BQ != nullptr) {
      delete BQ;
      BQ = nullptr;
    }
    return XRayLogInitStatus::XRAY_LOG_UNINITIALIZED;
  }

  static bool UNUSED Once = [] {
    pthread_key_create(&__xray_fdr_internal::Key, +[](void *) {
      auto &TLD = __xray_fdr_internal::getThreadLocalData();
      if (TLD.BQ == nullptr)
        return;
      auto EC = TLD.BQ->releaseBuffer(TLD.Buffer);
      if (EC != BufferQueue::ErrorCode::Ok)
        Report("At thread exit, failed to release buffer at %p; error=%s\n",
               TLD.Buffer.Buffer, BufferQueue::getErrorString(EC));
    });
    return false;
  }();

  // Arg1 handler should go in first to avoid concurrent code accidentally
  // falling back to arg0 when it should have ran arg1.
  __xray_set_handler_arg1(fdrLoggingHandleArg1);
  // Install the actual handleArg0 handler after initialising the buffers.
  __xray_set_handler(fdrLoggingHandleArg0);
  __xray_set_customevent_handler(fdrLoggingHandleCustomEvent);

  __sanitizer::atomic_store(&LoggingStatus,
                            XRayLogInitStatus::XRAY_LOG_INITIALIZED,
                            __sanitizer::memory_order_release);

  if (__sanitizer::Verbosity())
    Report("XRay FDR init successful.\n");
  return XRayLogInitStatus::XRAY_LOG_INITIALIZED;
}

} // namespace __xray

static auto UNUSED Unused = [] {
  using namespace __xray;
  if (flags()->xray_fdr_log) {
    XRayLogImpl Impl{
        fdrLoggingInit,
        fdrLoggingFinalize,
        fdrLoggingHandleArg0,
        fdrLoggingFlush,
    };
    __xray_set_log_impl(Impl);
  }
  return true;
}();
