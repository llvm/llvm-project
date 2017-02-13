//===-- xray_fdr_logging.cc ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of XRay, a dynamic runtime instruementation system.
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

namespace {
thread_local BufferQueue::Buffer Buffer;
thread_local char *RecordPtr = nullptr;

void setupNewBuffer(const BufferQueue::Buffer &Buffer) XRAY_NEVER_INSTRUMENT {
  RecordPtr = static_cast<char *>(Buffer.Buffer);

  static constexpr int InitRecordsCount = 2;
  std::aligned_storage<sizeof(MetadataRecord)>::type Records[InitRecordsCount];
  {
    // Write out a MetadataRecord to signify that this is the start of a new
    // buffer, associated with a particular thread, with a new CPU.  For the
    // data, we have 15 bytes to squeeze as much information as we can.  At this
    // point we only write down the following bytes:
    //   - Thread ID (pid_t, 4 bytes)
    auto &NewBuffer = *reinterpret_cast<MetadataRecord *>(&Records[0]);
    NewBuffer.Type = RecordType::Metadata;
    NewBuffer.RecordKind = MetadataRecord::RecordKinds::NewBuffer;
    pid_t Tid = syscall(SYS_gettid);
    std::memcpy(&NewBuffer.Data, &Tid, sizeof(pid_t));
  }

  // Also write the WalltimeMarker record.
  {
    static_assert(sizeof(time_t) <= 8, "time_t needs to be at most 8 bytes");
    auto &WalltimeMarker = *reinterpret_cast<MetadataRecord *>(&Records[1]);
    WalltimeMarker.Type = RecordType::Metadata;
    WalltimeMarker.RecordKind = MetadataRecord::RecordKinds::WalltimeMarker;
    timespec TS{0, 0};
    clock_gettime(CLOCK_MONOTONIC, &TS);

    // We only really need microsecond precision here, and enforce across
    // platforms that we need 64-bit seconds and 32-bit microseconds encoded in
    // the Metadata record.
    int32_t Micros = TS.tv_nsec / 1000;
    int64_t Seconds = TS.tv_sec;
    std::memcpy(WalltimeMarker.Data, &Seconds, sizeof(Seconds));
    std::memcpy(WalltimeMarker.Data + sizeof(Seconds), &Micros, sizeof(Micros));
  }
  std::memcpy(RecordPtr, Records, sizeof(MetadataRecord) * InitRecordsCount);
  RecordPtr += sizeof(MetadataRecord) * InitRecordsCount;
}

void writeNewCPUIdMetadata(uint16_t CPU, uint64_t TSC) XRAY_NEVER_INSTRUMENT {
  MetadataRecord NewCPUId;
  NewCPUId.Type = RecordType::Metadata;
  NewCPUId.RecordKind = MetadataRecord::RecordKinds::NewCPUId;

  // The data for the New CPU will contain the following bytes:
  //   - CPU ID (uint16_t, 2 bytes)
  //   - Full TSC (uint64_t, 8 bytes)
  // Total = 12 bytes.
  std::memcpy(&NewCPUId.Data, &CPU, sizeof(CPU));
  std::memcpy(&NewCPUId.Data[sizeof(CPU)], &TSC, sizeof(TSC));
  std::memcpy(RecordPtr, &NewCPUId, sizeof(MetadataRecord));
  RecordPtr += sizeof(MetadataRecord);
}

void writeEOBMetadata() XRAY_NEVER_INSTRUMENT {
  MetadataRecord EOBMeta;
  EOBMeta.Type = RecordType::Metadata;
  EOBMeta.RecordKind = MetadataRecord::RecordKinds::EndOfBuffer;
  // For now we don't write any bytes into the Data field.
  std::memcpy(RecordPtr, &EOBMeta, sizeof(MetadataRecord));
  RecordPtr += sizeof(MetadataRecord);
}

void writeTSCWrapMetadata(uint64_t TSC) XRAY_NEVER_INSTRUMENT {
  MetadataRecord TSCWrap;
  TSCWrap.Type = RecordType::Metadata;
  TSCWrap.RecordKind = MetadataRecord::RecordKinds::TSCWrap;

  // The data for the TSCWrap record contains the following bytes:
  //   - Full TSC (uint64_t, 8 bytes)
  // Total = 8 bytes.
  std::memcpy(&TSCWrap.Data, &TSC, sizeof(TSC));
  std::memcpy(RecordPtr, &TSCWrap, sizeof(MetadataRecord));
  RecordPtr += sizeof(MetadataRecord);
}

constexpr auto MetadataRecSize = sizeof(MetadataRecord);
constexpr auto FunctionRecSize = sizeof(FunctionRecord);

class ThreadExitBufferCleanup {
  std::weak_ptr<BufferQueue> Buffers;
  BufferQueue::Buffer &Buffer;

public:
  explicit ThreadExitBufferCleanup(std::weak_ptr<BufferQueue> BQ,
                                   BufferQueue::Buffer &Buffer)
      XRAY_NEVER_INSTRUMENT : Buffers(BQ),
                              Buffer(Buffer) {}

  ~ThreadExitBufferCleanup() noexcept XRAY_NEVER_INSTRUMENT {
    if (RecordPtr == nullptr)
      return;

    // We make sure that upon exit, a thread will write out the EOB
    // MetadataRecord in the thread-local log, and also release the buffer to
    // the queue.
    assert((RecordPtr + MetadataRecSize) - static_cast<char *>(Buffer.Buffer) >=
           static_cast<ptrdiff_t>(MetadataRecSize));
    if (auto BQ = Buffers.lock()) {
      writeEOBMetadata();
      if (auto EC = BQ->releaseBuffer(Buffer))
        Report("Failed to release buffer at %p; error=%s\n", Buffer.Buffer,
               EC.message().c_str());
      return;
    }
  }
};

class RecursionGuard {
  bool &Running;
  const bool Valid;

public:
  explicit RecursionGuard(bool &R) : Running(R), Valid(!R) {
    if (Valid)
      Running = true;
  }

  RecursionGuard(const RecursionGuard &) = delete;
  RecursionGuard(RecursionGuard &&) = delete;
  RecursionGuard &operator=(const RecursionGuard &) = delete;
  RecursionGuard &operator=(RecursionGuard &&) = delete;

  explicit operator bool() const { return Valid; }

  ~RecursionGuard() noexcept {
    if (Valid)
      Running = false;
  }
};

inline bool loggingInitialized() {
  return LoggingStatus.load(std::memory_order_acquire) ==
         XRayLogInitStatus::XRAY_LOG_INITIALIZED;
}

} // namespace

void fdrLoggingHandleArg0(int32_t FuncId,
                           XRayEntryType Entry) XRAY_NEVER_INSTRUMENT {
  // We want to get the TSC as early as possible, so that we can check whether
  // we've seen this CPU before. We also do it before we load anything else, to
  // allow for forward progress with the scheduling.
  unsigned char CPU;
  uint64_t TSC = __xray::readTSC(CPU);

  // Bail out right away if logging is not initialized yet.
  if (LoggingStatus.load(std::memory_order_acquire) !=
      XRayLogInitStatus::XRAY_LOG_INITIALIZED)
    return;

  // We use a thread_local variable to keep track of which CPUs we've already
  // run, and the TSC times for these CPUs. This allows us to stop repeating the
  // CPU field in the function records.
  //
  // We assume that we'll support only 65536 CPUs for x86_64.
  thread_local uint16_t CurrentCPU = std::numeric_limits<uint16_t>::max();
  thread_local uint64_t LastTSC = 0;

  // Make sure a thread that's ever called handleArg0 has a thread-local
  // live reference to the buffer queue for this particular instance of
  // FDRLogging, and that we're going to clean it up when the thread exits.
  thread_local auto LocalBQ = BQ;
  thread_local ThreadExitBufferCleanup Cleanup(LocalBQ, Buffer);

  // Prevent signal handler recursion, so in case we're already in a log writing
  // mode and the signal handler comes in (and is also instrumented) then we
  // don't want to be clobbering potentially partial writes already happening in
  // the thread. We use a simple thread_local latch to only allow one on-going
  // handleArg0 to happen at any given time.
  thread_local bool Running = false;
  RecursionGuard Guard{Running};
  if (!Guard) {
    assert(Running == true && "RecursionGuard is buggy!");
    return;
  }

  if (!loggingInitialized() || LocalBQ->finalizing()) {
    writeEOBMetadata();
    if (auto EC = BQ->releaseBuffer(Buffer)) {
      Report("Failed to release buffer at %p; error=%s\n", Buffer.Buffer,
             EC.message().c_str());
      return;
    }
    RecordPtr = nullptr;
  }

  if (Buffer.Buffer == nullptr) {
    if (auto EC = LocalBQ->getBuffer(Buffer)) {
      auto LS = LoggingStatus.load(std::memory_order_acquire);
      if (LS != XRayLogInitStatus::XRAY_LOG_FINALIZING &&
          LS != XRayLogInitStatus::XRAY_LOG_FINALIZED)
        Report("Failed to acquire a buffer; error=%s\n", EC.message().c_str());
      return;
    }

    setupNewBuffer(Buffer);
  }

  if (CurrentCPU == std::numeric_limits<uint16_t>::max()) {
    // This means this is the first CPU this thread has ever run on. We set the
    // current CPU and record this as the first TSC we've seen.
    CurrentCPU = CPU;
    writeNewCPUIdMetadata(CPU, TSC);
  }

  // Before we go setting up writing new function entries, we need to be really
  // careful about the pointer math we're doing. This means we need to ensure
  // that the record we are about to write is going to fit into the buffer,
  // without overflowing the buffer.
  //
  // To do this properly, we use the following assumptions:
  //
  //   - The least number of bytes we will ever write is 8
  //     (sizeof(FunctionRecord)) only if the delta between the previous entry
  //     and this entry is within 32 bits.
  //   - The most number of bytes we will ever write is 8 + 16 = 24. This is
  //     computed by:
  //
  //       sizeof(FunctionRecord) + sizeof(MetadataRecord)
  //
  //     These arise in the following cases:
  //
  //       1. When the delta between the TSC we get and the previous TSC for the
  //          same CPU is outside of the uint32_t range, we end up having to
  //          write a MetadataRecord to indicate a "tsc wrap" before the actual
  //          FunctionRecord.
  //       2. When we learn that we've moved CPUs, we need to write a
  //          MetadataRecord to indicate a "cpu change", and thus write out the
  //          current TSC for that CPU before writing out the actual
  //          FunctionRecord.
  //       3. When we learn about a new CPU ID, we need to write down a "new cpu
  //          id" MetadataRecord before writing out the actual FunctionRecord.
  //
  //   - An End-of-Buffer (EOB) MetadataRecord is 16 bytes.
  //
  // So the math we need to do is to determine whether writing 24 bytes past the
  // current pointer leaves us with enough bytes to write the EOB
  // MetadataRecord. If we don't have enough space after writing as much as 24
  // bytes in the end of the buffer, we need to write out the EOB, get a new
  // Buffer, set it up properly before doing any further writing.
  //
  char *BufferStart = static_cast<char *>(Buffer.Buffer);
  if ((RecordPtr + (MetadataRecSize + FunctionRecSize)) - BufferStart <
      static_cast<ptrdiff_t>(MetadataRecSize)) {
    writeEOBMetadata();
    if (auto EC = LocalBQ->releaseBuffer(Buffer)) {
      Report("Failed to release buffer at %p; error=%s\n", Buffer.Buffer,
             EC.message().c_str());
      return;
    }
    if (auto EC = LocalBQ->getBuffer(Buffer)) {
      Report("Failed to acquire a buffer; error=%s\n", EC.message().c_str());
      return;
    }
    setupNewBuffer(Buffer);
  }

  // By this point, we are now ready to write at most 24 bytes (one metadata
  // record and one function record).
  BufferStart = static_cast<char *>(Buffer.Buffer);
  assert((RecordPtr + (MetadataRecSize + FunctionRecSize)) - BufferStart >=
             static_cast<ptrdiff_t>(MetadataRecSize) &&
         "Misconfigured BufferQueue provided; Buffer size not large enough.");

  std::aligned_storage<sizeof(FunctionRecord), alignof(FunctionRecord)>::type
      AlignedFuncRecordBuffer;
  auto &FuncRecord =
      *reinterpret_cast<FunctionRecord *>(&AlignedFuncRecordBuffer);
  FuncRecord.Type = RecordType::Function;

  // Only get the lower 28 bits of the function id.
  FuncRecord.FuncId = FuncId & ~(0x0F << 28);

  // Here we compute the TSC Delta. There are a few interesting situations we
  // need to account for:
  //
  //   - The thread has migrated to a different CPU. If this is the case, then
  //     we write down the following records:
  //
  //       1. A 'NewCPUId' Metadata record.
  //       2. A FunctionRecord with a 0 for the TSCDelta field.
  //
  //   - The TSC delta is greater than the 32 bits we can store in a
  //     FunctionRecord. In this case we write down the following records:
  //
  //       1. A 'TSCWrap' Metadata record.
  //       2. A FunctionRecord with a 0 for the TSCDelta field.
  //
  //   - The TSC delta is representable within the 32 bits we can store in a
  //     FunctionRecord. In this case we write down just a FunctionRecord with
  //     the correct TSC delta.
  //
  FuncRecord.TSCDelta = 0;
  if (CPU != CurrentCPU) {
    // We've moved to a new CPU.
    writeNewCPUIdMetadata(CPU, TSC);
  } else {
    // If the delta is greater than the range for a uint32_t, then we write out
    // the TSC wrap metadata entry with the full TSC, and the TSC for the
    // function record be 0.
    auto Delta = LastTSC - TSC;
    if (Delta > (1ULL << 32) - 1)
      writeTSCWrapMetadata(TSC);
    else
      FuncRecord.TSCDelta = Delta;
  }

  // We then update our "LastTSC" and "CurrentCPU" thread-local variables to aid
  // us in future computations of this TSC delta value.
  LastTSC = TSC;
  CurrentCPU = CPU;

  switch (Entry) {
  case XRayEntryType::ENTRY:
    FuncRecord.RecordKind = FunctionRecord::RecordKinds::FunctionEnter;
    break;
  case XRayEntryType::EXIT:
    FuncRecord.RecordKind = FunctionRecord::RecordKinds::FunctionExit;
    break;
  case XRayEntryType::TAIL:
    FuncRecord.RecordKind = FunctionRecord::RecordKinds::FunctionTailExit;
    break;
  }

  std::memcpy(RecordPtr, &AlignedFuncRecordBuffer, sizeof(FunctionRecord));
  RecordPtr += sizeof(FunctionRecord);

  // If we've exhausted the buffer by this time, we then release the buffer to
  // make sure that other threads may start using this buffer.
  if ((RecordPtr + MetadataRecSize) - BufferStart == MetadataRecSize) {
    writeEOBMetadata();
    if (auto EC = LocalBQ->releaseBuffer(Buffer)) {
      Report("Failed releasing buffer at %p; error=%s\n", Buffer.Buffer,
             EC.message().c_str());
      return;
    }
    RecordPtr = nullptr;
  }
}

} // namespace __xray

static auto Unused = [] {
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
