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
#include <cassert>
#include <cstring>
#include <memory>
#include <sys/syscall.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <unordered_map>

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
std::shared_ptr<BufferQueue> BQ;

std::atomic<XRayLogInitStatus> LoggingStatus{
    XRayLogInitStatus::XRAY_LOG_UNINITIALIZED};

std::atomic<XRayLogFlushStatus> LogFlushStatus{
    XRayLogFlushStatus::XRAY_LOG_NOT_FLUSHING};

std::unique_ptr<FDRLoggingOptions> FDROptions;

XRayLogInitStatus fdrLoggingInit(std::size_t BufferSize, std::size_t BufferMax,
                                 void *Options,
                                 size_t OptionsSize) XRAY_NEVER_INSTRUMENT {
  assert(OptionsSize == sizeof(FDRLoggingOptions));
  XRayLogInitStatus CurrentStatus = XRayLogInitStatus::XRAY_LOG_UNINITIALIZED;
  if (!LoggingStatus.compare_exchange_strong(
          CurrentStatus, XRayLogInitStatus::XRAY_LOG_INITIALIZING,
          std::memory_order_release, std::memory_order_relaxed))
    return CurrentStatus;

  FDROptions.reset(new FDRLoggingOptions());
  *FDROptions = *reinterpret_cast<FDRLoggingOptions *>(Options);
  if (FDROptions->ReportErrors)
    SetPrintfAndReportCallback(printToStdErr);

  bool Success = false;
  BQ = std::make_shared<BufferQueue>(BufferSize, BufferMax, Success);
  if (!Success) {
    Report("BufferQueue init failed.\n");
    return XRayLogInitStatus::XRAY_LOG_UNINITIALIZED;
  }

  // Install the actual handleArg0 handler after initialising the buffers.
  __xray_set_handler(fdrLoggingHandleArg0);

  LoggingStatus.store(XRayLogInitStatus::XRAY_LOG_INITIALIZED,
                      std::memory_order_release);
  return XRayLogInitStatus::XRAY_LOG_INITIALIZED;
}

// Must finalize before flushing.
XRayLogFlushStatus fdrLoggingFlush() XRAY_NEVER_INSTRUMENT {
  if (LoggingStatus.load(std::memory_order_acquire) !=
      XRayLogInitStatus::XRAY_LOG_FINALIZED)
    return XRayLogFlushStatus::XRAY_LOG_NOT_FLUSHING;

  XRayLogFlushStatus Result = XRayLogFlushStatus::XRAY_LOG_NOT_FLUSHING;
  if (!LogFlushStatus.compare_exchange_strong(
          Result, XRayLogFlushStatus::XRAY_LOG_FLUSHING,
          std::memory_order_release, std::memory_order_relaxed))
    return Result;

  // Make a copy of the BufferQueue pointer to prevent other threads that may be
  // resetting it from blowing away the queue prematurely while we're dealing
  // with it.
  auto LocalBQ = BQ;

  // We write out the file in the following format:
  //
  //   1) We write down the XRay file header with version 1, type FDR_LOG.
  //   2) Then we use the 'apply' member of the BufferQueue that's live, to
  //      ensure that at this point in time we write down the buffers that have
  //      been released (and marked "used") -- we dump the full buffer for now
  //      (fixed-sized) and let the tools reading the buffers deal with the data
  //      afterwards.
  //
  int Fd = FDROptions->Fd;
  if (Fd == -1)
    Fd = getLogFD();
  if (Fd == -1) {
    auto Result = XRayLogFlushStatus::XRAY_LOG_NOT_FLUSHING;
    LogFlushStatus.store(Result, std::memory_order_release);
    return Result;
  }

  XRayFileHeader Header;
  Header.Version = 1;
  Header.Type = FileTypes::FDR_LOG;
  Header.CycleFrequency = getTSCFrequency();
  // FIXME: Actually check whether we have 'constant_tsc' and 'nonstop_tsc'
  // before setting the values in the header.
  Header.ConstantTSC = 1;
  Header.NonstopTSC = 1;
  clock_gettime(CLOCK_REALTIME, &Header.TS);
  retryingWriteAll(Fd, reinterpret_cast<char *>(&Header),
                   reinterpret_cast<char *>(&Header) + sizeof(Header));
  LocalBQ->apply([&](const BufferQueue::Buffer &B) {
    retryingWriteAll(Fd, reinterpret_cast<char *>(B.Buffer),
                     reinterpret_cast<char *>(B.Buffer) + B.Size);
  });
  LogFlushStatus.store(XRayLogFlushStatus::XRAY_LOG_FLUSHED,
                       std::memory_order_release);
  return XRayLogFlushStatus::XRAY_LOG_FLUSHED;
}

XRayLogInitStatus fdrLoggingFinalize() XRAY_NEVER_INSTRUMENT {
  XRayLogInitStatus CurrentStatus = XRayLogInitStatus::XRAY_LOG_INITIALIZED;
  if (!LoggingStatus.compare_exchange_strong(
          CurrentStatus, XRayLogInitStatus::XRAY_LOG_FINALIZING,
          std::memory_order_release, std::memory_order_relaxed))
    return CurrentStatus;

  // Do special things to make the log finalize itself, and not allow any more
  // operations to be performed until re-initialized.
  BQ->finalize();

  LoggingStatus.store(XRayLogInitStatus::XRAY_LOG_FINALIZED,
                      std::memory_order_release);
  return XRayLogInitStatus::XRAY_LOG_FINALIZED;
}

XRayLogInitStatus fdrLoggingReset() XRAY_NEVER_INSTRUMENT {
  XRayLogInitStatus CurrentStatus = XRayLogInitStatus::XRAY_LOG_FINALIZED;
  if (!LoggingStatus.compare_exchange_strong(
          CurrentStatus, XRayLogInitStatus::XRAY_LOG_UNINITIALIZED,
          std::memory_order_release, std::memory_order_relaxed))
    return CurrentStatus;

  // Release the in-memory buffer queue.
  BQ.reset();

  // Spin until the flushing status is flushed.
  XRayLogFlushStatus CurrentFlushingStatus =
      XRayLogFlushStatus::XRAY_LOG_FLUSHED;
  while (!LogFlushStatus.compare_exchange_weak(
      CurrentFlushingStatus, XRayLogFlushStatus::XRAY_LOG_NOT_FLUSHING,
      std::memory_order_release, std::memory_order_relaxed)) {
    if (CurrentFlushingStatus == XRayLogFlushStatus::XRAY_LOG_NOT_FLUSHING)
      break;
    CurrentFlushingStatus = XRayLogFlushStatus::XRAY_LOG_FLUSHED;
  }

  // At this point, we know that the status is flushed, and that we can assume
  return XRayLogInitStatus::XRAY_LOG_UNINITIALIZED;
}

void fdrLoggingHandleArg0(int32_t FuncId,
                          XRayEntryType Entry) XRAY_NEVER_INSTRUMENT {
  // We want to get the TSC as early as possible, so that we can check whether
  // we've seen this CPU before. We also do it before we load anything else, to
  // allow for forward progress with the scheduling.
  unsigned char CPU;
  uint64_t TSC = __xray::readTSC(CPU);

  __xray_fdr_internal::processFunctionHook(FuncId, Entry, TSC, CPU,
                                           clock_gettime, LoggingStatus, BQ);
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
