//===-- xray_fdr_logging_impl.h ---------------------------------*- C++ -*-===//
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
// Here we implement the thread local state management and record i/o for Flight
// Data Recorder mode for XRay, where we use compact structures to store records
// in memory as well as when writing out the data to files.
//
//===----------------------------------------------------------------------===//
#ifndef XRAY_XRAY_FDR_LOGGING_IMPL_H
#define XRAY_XRAY_FDR_LOGGING_IMPL_H

#include <cassert>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#include "sanitizer_common/sanitizer_common.h"
#include "xray/xray_log_interface.h"
#include "xray_buffer_queue.h"
#include "xray_defs.h"
#include "xray_fdr_log_records.h"
#include "xray_flags.h"
#include "xray_tsc.h"

namespace __xray {

__sanitizer::atomic_sint32_t LoggingStatus = {
    XRayLogInitStatus::XRAY_LOG_UNINITIALIZED};

/// We expose some of the state transitions when FDR logging mode is operating
/// such that we can simulate a series of log events that may occur without
/// and test with determinism without worrying about the real CPU time.
///
/// Because the code uses thread_local allocation extensively as part of its
/// design, callers that wish to test events occuring on different threads
/// will actually have to run them on different threads.
///
/// This also means that it is possible to break invariants maintained by
/// cooperation with xray_fdr_logging class, so be careful and think twice.
namespace __xray_fdr_internal {

/// Writes the new buffer record and wallclock time that begin a buffer for a
/// thread to MemPtr and increments MemPtr. Bypasses the thread local state
/// machine and writes directly to memory without checks.
static void writeNewBufferPreamble(pid_t Tid, timespec TS, char *&MemPtr);

/// Write a metadata record to switch to a new CPU to MemPtr and increments
/// MemPtr. Bypasses the thread local state machine and writes directly to
/// memory without checks.
static void writeNewCPUIdMetadata(uint16_t CPU, uint64_t TSC, char *&MemPtr);

/// Writes an EOB metadata record to MemPtr and increments MemPtr. Bypasses the
/// thread local state machine and writes directly to memory without checks.
static void writeEOBMetadata(char *&MemPtr);

/// Writes a TSC Wrap metadata record to MemPtr and increments MemPtr. Bypasses
/// the thread local state machine and directly writes to memory without checks.
static void writeTSCWrapMetadata(uint64_t TSC, char *&MemPtr);

/// Writes a Function Record to MemPtr and increments MemPtr. Bypasses the
/// thread local state machine and writes the function record directly to
/// memory.
static void writeFunctionRecord(int FuncId, uint32_t TSCDelta,
                                XRayEntryType EntryType, char *&MemPtr);

/// Sets up a new buffer in thread_local storage and writes a preamble. The
/// wall_clock_reader function is used to populate the WallTimeRecord entry.
static void setupNewBuffer(int (*wall_clock_reader)(clockid_t,
                                                    struct timespec *));

/// Called to record CPU time for a new CPU within the current thread.
static void writeNewCPUIdMetadata(uint16_t CPU, uint64_t TSC);

/// Called to close the buffer when the thread exhausts the buffer or when the
/// thread exits (via a thread local variable destructor).
static void writeEOBMetadata();

/// TSC Wrap records are written when a TSC delta encoding scheme overflows.
static void writeTSCWrapMetadata(uint64_t TSC);

/// Here's where the meat of the processing happens. The writer captures
/// function entry, exit and tail exit points with a time and will create
/// TSCWrap, NewCPUId and Function records as necessary. The writer might
/// walk backward through its buffer and erase trivial functions to avoid
/// polluting the log and may use the buffer queue to obtain or release a
/// buffer.
static void processFunctionHook(int32_t FuncId, XRayEntryType Entry,
                                uint64_t TSC, unsigned char CPU,
                                int (*wall_clock_reader)(clockid_t,
                                                         struct timespec *),
                                __sanitizer::atomic_sint32_t &LoggingStatus,
                                const std::shared_ptr<BufferQueue> &BQ);

//-----------------------------------------------------------------------------|
// The rest of the file is implementation.                                     |
//-----------------------------------------------------------------------------|
// Functions are implemented in the header for inlining since we don't want    |
// to grow the stack when we've hijacked the binary for logging.               |
//-----------------------------------------------------------------------------|

namespace {

thread_local BufferQueue::Buffer Buffer;
thread_local char *RecordPtr = nullptr;

// The number of FunctionEntry records immediately preceding RecordPtr.
thread_local uint8_t NumConsecutiveFnEnters = 0;

// The number of adjacent, consecutive pairs of FunctionEntry, Tail Exit
// records preceding RecordPtr.
thread_local uint8_t NumTailCalls = 0;

constexpr auto MetadataRecSize = sizeof(MetadataRecord);
constexpr auto FunctionRecSize = sizeof(FunctionRecord);

// We use a thread_local variable to keep track of which CPUs we've already
// run, and the TSC times for these CPUs. This allows us to stop repeating the
// CPU field in the function records.
//
// We assume that we'll support only 65536 CPUs for x86_64.
thread_local uint16_t CurrentCPU = std::numeric_limits<uint16_t>::max();
thread_local uint64_t LastTSC = 0;
thread_local uint64_t LastFunctionEntryTSC = 0;

class ThreadExitBufferCleanup {
  std::shared_ptr<BufferQueue> &Buffers;
  BufferQueue::Buffer &Buffer;

public:
  explicit ThreadExitBufferCleanup(std::shared_ptr<BufferQueue> &BQ,
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
    if (Buffers) {
      writeEOBMetadata();
      auto EC = Buffers->releaseBuffer(Buffer);
      if (EC != BufferQueue::ErrorCode::Ok)
        Report("Failed to release buffer at %p; error=%s\n", Buffer.Buffer,
               BufferQueue::getErrorString(EC));
      Buffers = nullptr;
      return;
    }
  }
};

// Make sure a thread that's ever called handleArg0 has a thread-local
// live reference to the buffer queue for this particular instance of
// FDRLogging, and that we're going to clean it up when the thread exits.
thread_local std::shared_ptr<BufferQueue> LocalBQ = nullptr;
thread_local ThreadExitBufferCleanup Cleanup(LocalBQ, Buffer);

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

inline bool loggingInitialized(
    const __sanitizer::atomic_sint32_t &LoggingStatus) XRAY_NEVER_INSTRUMENT {
  return __sanitizer::atomic_load(&LoggingStatus,
                                  __sanitizer::memory_order_acquire) ==
         XRayLogInitStatus::XRAY_LOG_INITIALIZED;
}

} // namespace

inline void writeNewBufferPreamble(pid_t Tid, timespec TS,
                                   char *&MemPtr) XRAY_NEVER_INSTRUMENT {
  static constexpr int InitRecordsCount = 2;
  std::aligned_storage<sizeof(MetadataRecord)>::type Records[InitRecordsCount];
  {
    // Write out a MetadataRecord to signify that this is the start of a new
    // buffer, associated with a particular thread, with a new CPU.  For the
    // data, we have 15 bytes to squeeze as much information as we can.  At this
    // point we only write down the following bytes:
    //   - Thread ID (pid_t, 4 bytes)
    auto &NewBuffer = *reinterpret_cast<MetadataRecord *>(&Records[0]);
    NewBuffer.Type = uint8_t(RecordType::Metadata);
    NewBuffer.RecordKind = uint8_t(MetadataRecord::RecordKinds::NewBuffer);
    std::memcpy(&NewBuffer.Data, &Tid, sizeof(pid_t));
  }
  // Also write the WalltimeMarker record.
  {
    static_assert(sizeof(time_t) <= 8, "time_t needs to be at most 8 bytes");
    auto &WalltimeMarker = *reinterpret_cast<MetadataRecord *>(&Records[1]);
    WalltimeMarker.Type = uint8_t(RecordType::Metadata);
    WalltimeMarker.RecordKind =
        uint8_t(MetadataRecord::RecordKinds::WalltimeMarker);

    // We only really need microsecond precision here, and enforce across
    // platforms that we need 64-bit seconds and 32-bit microseconds encoded in
    // the Metadata record.
    int32_t Micros = TS.tv_nsec / 1000;
    int64_t Seconds = TS.tv_sec;
    std::memcpy(WalltimeMarker.Data, &Seconds, sizeof(Seconds));
    std::memcpy(WalltimeMarker.Data + sizeof(Seconds), &Micros, sizeof(Micros));
  }
  std::memcpy(MemPtr, Records, sizeof(MetadataRecord) * InitRecordsCount);
  MemPtr += sizeof(MetadataRecord) * InitRecordsCount;
  NumConsecutiveFnEnters = 0;
  NumTailCalls = 0;
}

inline void setupNewBuffer(int (*wall_clock_reader)(
    clockid_t, struct timespec *)) XRAY_NEVER_INSTRUMENT {
  RecordPtr = static_cast<char *>(Buffer.Buffer);
  pid_t Tid = syscall(SYS_gettid);
  timespec TS{0, 0};
  // This is typically clock_gettime, but callers have injection ability.
  wall_clock_reader(CLOCK_MONOTONIC, &TS);
  writeNewBufferPreamble(Tid, TS, RecordPtr);
  NumConsecutiveFnEnters = 0;
  NumTailCalls = 0;
}

inline void writeNewCPUIdMetadata(uint16_t CPU, uint64_t TSC,
                                  char *&MemPtr) XRAY_NEVER_INSTRUMENT {
  MetadataRecord NewCPUId;
  NewCPUId.Type = uint8_t(RecordType::Metadata);
  NewCPUId.RecordKind = uint8_t(MetadataRecord::RecordKinds::NewCPUId);

  // The data for the New CPU will contain the following bytes:
  //   - CPU ID (uint16_t, 2 bytes)
  //   - Full TSC (uint64_t, 8 bytes)
  // Total = 10 bytes.
  std::memcpy(&NewCPUId.Data, &CPU, sizeof(CPU));
  std::memcpy(&NewCPUId.Data[sizeof(CPU)], &TSC, sizeof(TSC));
  std::memcpy(MemPtr, &NewCPUId, sizeof(MetadataRecord));
  MemPtr += sizeof(MetadataRecord);
  NumConsecutiveFnEnters = 0;
  NumTailCalls = 0;
}

inline void writeNewCPUIdMetadata(uint16_t CPU,
                                  uint64_t TSC) XRAY_NEVER_INSTRUMENT {
  writeNewCPUIdMetadata(CPU, TSC, RecordPtr);
}

inline void writeEOBMetadata(char *&MemPtr) XRAY_NEVER_INSTRUMENT {
  MetadataRecord EOBMeta;
  EOBMeta.Type = uint8_t(RecordType::Metadata);
  EOBMeta.RecordKind = uint8_t(MetadataRecord::RecordKinds::EndOfBuffer);
  // For now we don't write any bytes into the Data field.
  std::memcpy(MemPtr, &EOBMeta, sizeof(MetadataRecord));
  MemPtr += sizeof(MetadataRecord);
  NumConsecutiveFnEnters = 0;
  NumTailCalls = 0;
}

inline void writeEOBMetadata() XRAY_NEVER_INSTRUMENT {
  writeEOBMetadata(RecordPtr);
}

inline void writeTSCWrapMetadata(uint64_t TSC,
                                 char *&MemPtr) XRAY_NEVER_INSTRUMENT {
  MetadataRecord TSCWrap;
  TSCWrap.Type = uint8_t(RecordType::Metadata);
  TSCWrap.RecordKind = uint8_t(MetadataRecord::RecordKinds::TSCWrap);

  // The data for the TSCWrap record contains the following bytes:
  //   - Full TSC (uint64_t, 8 bytes)
  // Total = 8 bytes.
  std::memcpy(&TSCWrap.Data, &TSC, sizeof(TSC));
  std::memcpy(MemPtr, &TSCWrap, sizeof(MetadataRecord));
  MemPtr += sizeof(MetadataRecord);
  NumConsecutiveFnEnters = 0;
  NumTailCalls = 0;
}

inline void writeTSCWrapMetadata(uint64_t TSC) XRAY_NEVER_INSTRUMENT {
  writeTSCWrapMetadata(TSC, RecordPtr);
}

inline void writeFunctionRecord(int FuncId, uint32_t TSCDelta,
                                XRayEntryType EntryType,
                                char *&MemPtr) XRAY_NEVER_INSTRUMENT {
  std::aligned_storage<sizeof(FunctionRecord), alignof(FunctionRecord)>::type
      AlignedFuncRecordBuffer;
  auto &FuncRecord =
      *reinterpret_cast<FunctionRecord *>(&AlignedFuncRecordBuffer);
  FuncRecord.Type = uint8_t(RecordType::Function);
  // Only take 28 bits of the function id.
  FuncRecord.FuncId = FuncId & ~(0x0F << 28);
  FuncRecord.TSCDelta = TSCDelta;

  switch (EntryType) {
  case XRayEntryType::ENTRY:
    ++NumConsecutiveFnEnters;
    FuncRecord.RecordKind = uint8_t(FunctionRecord::RecordKinds::FunctionEnter);
    break;
  case XRayEntryType::LOG_ARGS_ENTRY:
    // We should not rewind functions with logged args.
    NumConsecutiveFnEnters = 0;
    NumTailCalls = 0;
    FuncRecord.RecordKind = uint8_t(FunctionRecord::RecordKinds::FunctionEnter);
    break;
  case XRayEntryType::EXIT:
    // If we've decided to log the function exit, we will never erase the log
    // before it.
    NumConsecutiveFnEnters = 0;
    NumTailCalls = 0;
    FuncRecord.RecordKind = uint8_t(FunctionRecord::RecordKinds::FunctionExit);
    break;
  case XRayEntryType::TAIL:
    // If we just entered the function we're tail exiting from or erased every
    // invocation since then, this function entry tail pair is a candidate to
    // be erased when the child function exits.
    if (NumConsecutiveFnEnters > 0) {
      ++NumTailCalls;
      NumConsecutiveFnEnters = 0;
    } else {
      // We will never be able to erase this tail call since we have logged
      // something in between the function entry and tail exit.
      NumTailCalls = 0;
      NumConsecutiveFnEnters = 0;
    }
    FuncRecord.RecordKind =
        uint8_t(FunctionRecord::RecordKinds::FunctionTailExit);
    break;
  case XRayEntryType::CUSTOM_EVENT: {
    // This is a bug in patching, so we'll report it once and move on.
    static bool Once = [&] {
      Report("Internal error: patched an XRay custom event call as a function; "
             "func id = %d\n",
             FuncId);
      return true;
    }();
    (void)Once;
    return;
  }
  }

  std::memcpy(MemPtr, &AlignedFuncRecordBuffer, sizeof(FunctionRecord));
  MemPtr += sizeof(FunctionRecord);
}

static uint64_t thresholdTicks() {
  static uint64_t TicksPerSec = probeRequiredCPUFeatures()
                                    ? getTSCFrequency()
                                    : __xray::NanosecondsPerSecond;
  static const uint64_t ThresholdTicks =
      TicksPerSec * flags()->xray_fdr_log_func_duration_threshold_us / 1000000;
  return ThresholdTicks;
}

// Re-point the thread local pointer into this thread's Buffer before the recent
// "Function Entry" record and any "Tail Call Exit" records after that.
static void rewindRecentCall(uint64_t TSC, uint64_t &LastTSC,
                             uint64_t &LastFunctionEntryTSC, int32_t FuncId) {
  using AlignedFuncStorage =
      std::aligned_storage<sizeof(FunctionRecord),
                           alignof(FunctionRecord)>::type;
  RecordPtr -= FunctionRecSize;
  AlignedFuncStorage AlignedFuncRecordBuffer;
  const auto &FuncRecord = *reinterpret_cast<FunctionRecord *>(
      std::memcpy(&AlignedFuncRecordBuffer, RecordPtr, FunctionRecSize));
  assert(FuncRecord.RecordKind ==
             uint8_t(FunctionRecord::RecordKinds::FunctionEnter) &&
         "Expected to find function entry recording when rewinding.");
  assert(FuncRecord.FuncId == (FuncId & ~(0x0F << 28)) &&
         "Expected matching function id when rewinding Exit");
  --NumConsecutiveFnEnters;
  LastTSC -= FuncRecord.TSCDelta;

  // We unwound one call. Update the state and return without writing a log.
  if (NumConsecutiveFnEnters != 0) {
    LastFunctionEntryTSC -= FuncRecord.TSCDelta;
    return;
  }

  // Otherwise we've rewound the stack of all function entries, we might be
  // able to rewind further by erasing tail call functions that are being
  // exited from via this exit.
  LastFunctionEntryTSC = 0;
  auto RewindingTSC = LastTSC;
  auto RewindingRecordPtr = RecordPtr - FunctionRecSize;
  while (NumTailCalls > 0) {
    AlignedFuncStorage TailExitRecordBuffer;
    // Rewind the TSC back over the TAIL EXIT record.
    const auto &ExpectedTailExit =
        *reinterpret_cast<FunctionRecord *>(std::memcpy(
            &TailExitRecordBuffer, RewindingRecordPtr, FunctionRecSize));

    assert(ExpectedTailExit.RecordKind ==
               uint8_t(FunctionRecord::RecordKinds::FunctionTailExit) &&
           "Expected to find tail exit when rewinding.");
    RewindingRecordPtr -= FunctionRecSize;
    RewindingTSC -= ExpectedTailExit.TSCDelta;
    AlignedFuncStorage FunctionEntryBuffer;
    const auto &ExpectedFunctionEntry = *reinterpret_cast<FunctionRecord *>(
        std::memcpy(&FunctionEntryBuffer, RewindingRecordPtr, FunctionRecSize));
    assert(ExpectedFunctionEntry.RecordKind ==
               uint8_t(FunctionRecord::RecordKinds::FunctionEnter) &&
           "Expected to find function entry when rewinding tail call.");
    assert(ExpectedFunctionEntry.FuncId == ExpectedTailExit.FuncId &&
           "Expected funcids to match when rewinding tail call.");

    // This tail call exceeded the threshold duration. It will not be erased.
    if ((TSC - RewindingTSC) >= thresholdTicks()) {
      NumTailCalls = 0;
      return;
    }

    // We can erase a tail exit pair that we're exiting through since
    // its duration is under threshold.
    --NumTailCalls;
    RewindingRecordPtr -= FunctionRecSize;
    RewindingTSC -= ExpectedFunctionEntry.TSCDelta;
    RecordPtr -= 2 * FunctionRecSize;
    LastTSC = RewindingTSC;
  }
}

inline bool releaseThreadLocalBuffer(BufferQueue *BQ) {
  auto EC = BQ->releaseBuffer(Buffer);
  if (EC != BufferQueue::ErrorCode::Ok) {
    Report("Failed to release buffer at %p; error=%s\n", Buffer.Buffer,
           BufferQueue::getErrorString(EC));
    return false;
  }
  return true;
}

inline bool prepareBuffer(int (*wall_clock_reader)(clockid_t,
                                                   struct timespec *),
                          size_t MaxSize) XRAY_NEVER_INSTRUMENT {
  char *BufferStart = static_cast<char *>(Buffer.Buffer);
  if ((RecordPtr + MaxSize) > (BufferStart + Buffer.Size - MetadataRecSize)) {
    writeEOBMetadata();
    if (!releaseThreadLocalBuffer(LocalBQ.get()))
      return false;
    auto EC = LocalBQ->getBuffer(Buffer);
    if (EC != BufferQueue::ErrorCode::Ok) {
      Report("Failed to acquire a buffer; error=%s\n",
             BufferQueue::getErrorString(EC));
      return false;
    }
    setupNewBuffer(wall_clock_reader);
  }
  return true;
}

inline bool isLogInitializedAndReady(
    std::shared_ptr<BufferQueue> &LocalBQ, uint64_t TSC, unsigned char CPU,
    int (*wall_clock_reader)(clockid_t,
                             struct timespec *)) XRAY_NEVER_INSTRUMENT {
  // Bail out right away if logging is not initialized yet.
  // We should take the opportunity to release the buffer though.
  auto Status = __sanitizer::atomic_load(&LoggingStatus,
                                         __sanitizer::memory_order_acquire);
  if (Status != XRayLogInitStatus::XRAY_LOG_INITIALIZED) {
    if (RecordPtr != nullptr &&
        (Status == XRayLogInitStatus::XRAY_LOG_FINALIZING ||
         Status == XRayLogInitStatus::XRAY_LOG_FINALIZED)) {
      writeEOBMetadata();
      if (!releaseThreadLocalBuffer(LocalBQ.get()))
        return false;
      RecordPtr = nullptr;
      LocalBQ = nullptr;
      return false;
    }
    return false;
  }

  if (!loggingInitialized(LoggingStatus) || LocalBQ->finalizing()) {
    writeEOBMetadata();
    if (!releaseThreadLocalBuffer(LocalBQ.get()))
      return false;
    RecordPtr = nullptr;
  }

  if (Buffer.Buffer == nullptr) {
    auto EC = LocalBQ->getBuffer(Buffer);
    if (EC != BufferQueue::ErrorCode::Ok) {
      auto LS = __sanitizer::atomic_load(&LoggingStatus,
                                         __sanitizer::memory_order_acquire);
      if (LS != XRayLogInitStatus::XRAY_LOG_FINALIZING &&
          LS != XRayLogInitStatus::XRAY_LOG_FINALIZED)
        Report("Failed to acquire a buffer; error=%s\n",
               BufferQueue::getErrorString(EC));
      return false;
    }

    setupNewBuffer(wall_clock_reader);
  }

  if (CurrentCPU == std::numeric_limits<uint16_t>::max()) {
    // This means this is the first CPU this thread has ever run on. We set
    // the current CPU and record this as the first TSC we've seen.
    CurrentCPU = CPU;
    writeNewCPUIdMetadata(CPU, TSC);
  }

  return true;
} // namespace __xray_fdr_internal

inline void endBufferIfFull() XRAY_NEVER_INSTRUMENT {
  auto BufferStart = static_cast<char *>(Buffer.Buffer);
  if ((RecordPtr + MetadataRecSize) - BufferStart == MetadataRecSize) {
    writeEOBMetadata();
    if (!releaseThreadLocalBuffer(LocalBQ.get()))
      return;
    RecordPtr = nullptr;
  }
}

inline void processFunctionHook(
    int32_t FuncId, XRayEntryType Entry, uint64_t TSC, unsigned char CPU,
    int (*wall_clock_reader)(clockid_t, struct timespec *),
    __sanitizer::atomic_sint32_t &LoggingStatus,
    const std::shared_ptr<BufferQueue> &BQ) XRAY_NEVER_INSTRUMENT {
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

  // In case the reference has been cleaned up before, we make sure we
  // initialize it to the provided BufferQueue.
  if (LocalBQ == nullptr)
    LocalBQ = BQ;

  if (!isLogInitializedAndReady(LocalBQ, TSC, CPU, wall_clock_reader))
    return;

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
  if (!prepareBuffer(wall_clock_reader, FunctionRecSize + MetadataRecSize)) {
    LocalBQ = nullptr;
    return;
  }

  // By this point, we are now ready to write at most 24 bytes (one metadata
  // record and one function record).
  assert((RecordPtr + (MetadataRecSize + FunctionRecSize)) -
                 static_cast<char *>(Buffer.Buffer) >=
             static_cast<ptrdiff_t>(MetadataRecSize) &&
         "Misconfigured BufferQueue provided; Buffer size not large enough.");

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
  uint32_t RecordTSCDelta = 0;
  if (CPU != CurrentCPU) {
    // We've moved to a new CPU.
    writeNewCPUIdMetadata(CPU, TSC);
  } else {
    // If the delta is greater than the range for a uint32_t, then we write out
    // the TSC wrap metadata entry with the full TSC, and the TSC for the
    // function record be 0.
    auto Delta = TSC - LastTSC;
    if (Delta > (1ULL << 32) - 1)
      writeTSCWrapMetadata(TSC);
    else
      RecordTSCDelta = Delta;
  }

  LastTSC = TSC;
  CurrentCPU = CPU;
  switch (Entry) {
  case XRayEntryType::ENTRY:
  case XRayEntryType::LOG_ARGS_ENTRY:
    // Update the thread local state for the next invocation.
    LastFunctionEntryTSC = TSC;
    break;
  case XRayEntryType::TAIL:
    break;
  case XRayEntryType::EXIT:
    // Break out and write the exit record if we can't erase any functions.
    if (NumConsecutiveFnEnters == 0 ||
        (TSC - LastFunctionEntryTSC) >= thresholdTicks())
      break;
    rewindRecentCall(TSC, LastTSC, LastFunctionEntryTSC, FuncId);
    return; // without writing log.
  case XRayEntryType::CUSTOM_EVENT: {
    // This is a bug in patching, so we'll report it once and move on.
    static bool Once = [&] {
      Report("Internal error: patched an XRay custom event call as a function; "
             "func id = %d",
             FuncId);
      return true;
    }();
    (void)Once;
    return;
  }
  }

  writeFunctionRecord(FuncId, RecordTSCDelta, Entry, RecordPtr);

  // If we've exhausted the buffer by this time, we then release the buffer to
  // make sure that other threads may start using this buffer.
  endBufferIfFull();
}

} // namespace __xray_fdr_internal
} // namespace __xray

#endif // XRAY_XRAY_FDR_LOGGING_IMPL_H
