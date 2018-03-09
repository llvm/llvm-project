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
#include <cstddef>
#include <cstring>
#include <limits>
#include <pthread.h>
#include <sys/syscall.h>
#include <time.h>
#include <type_traits>
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

/// Writes the new buffer record and wallclock time that begin a buffer for the
/// current thread.
static void writeNewBufferPreamble(pid_t Tid, timespec TS);

/// Writes a Function Record to the buffer associated with the current thread.
static void writeFunctionRecord(int FuncId, uint32_t TSCDelta,
                                XRayEntryType EntryType);

/// Sets up a new buffer in thread_local storage and writes a preamble. The
/// wall_clock_reader function is used to populate the WallTimeRecord entry.
static void setupNewBuffer(int (*wall_clock_reader)(clockid_t,
                                                    struct timespec *));

/// TSC Wrap records are written when a TSC delta encoding scheme overflows.
static void writeTSCWrapMetadata(uint64_t TSC);

// Group together thread-local-data in a struct, then hide it behind a function
// call so that it can be initialized on first use instead of as a global. We
// force the alignment to 64-bytes for x86 cache line alignment, as this
// structure is used in the hot path of implementation.
struct alignas(64) ThreadLocalData {
  BufferQueue::Buffer Buffer;
  char *RecordPtr = nullptr;
  // The number of FunctionEntry records immediately preceding RecordPtr.
  uint8_t NumConsecutiveFnEnters = 0;

  // The number of adjacent, consecutive pairs of FunctionEntry, Tail Exit
  // records preceding RecordPtr.
  uint8_t NumTailCalls = 0;

  // We use a thread_local variable to keep track of which CPUs we've already
  // run, and the TSC times for these CPUs. This allows us to stop repeating the
  // CPU field in the function records.
  //
  // We assume that we'll support only 65536 CPUs for x86_64.
  uint16_t CurrentCPU = std::numeric_limits<uint16_t>::max();
  uint64_t LastTSC = 0;
  uint64_t LastFunctionEntryTSC = 0;

  // Make sure a thread that's ever called handleArg0 has a thread-local
  // live reference to the buffer queue for this particular instance of
  // FDRLogging, and that we're going to clean it up when the thread exits.
  BufferQueue *BQ = nullptr;
};

static_assert(std::is_trivially_destructible<ThreadLocalData>::value,
              "ThreadLocalData must be trivially destructible");

static constexpr auto MetadataRecSize = sizeof(MetadataRecord);
static constexpr auto FunctionRecSize = sizeof(FunctionRecord);

// Use a global pthread key to identify thread-local data for logging.
static pthread_key_t Key;

// This function will initialize the thread-local data structure used by the FDR
// logging implementation and return a reference to it. The implementation
// details require a bit of care to maintain.
//
// First, some requirements on the implementation in general:
//
//   - XRay handlers should not call any memory allocation routines that may
//     delegate to an instrumented implementation. This means functions like
//     malloc() and free() should not be called while instrumenting.
//
//   - We would like to use some thread-local data initialized on first-use of
//     the XRay instrumentation. These allow us to implement unsynchronized
//     routines that access resources associated with the thread.
//
// The implementation here uses a few mechanisms that allow us to provide both
// the requirements listed above. We do this by:
//
//   1. Using a thread-local aligned storage buffer for representing the
//      ThreadLocalData struct. This data will be uninitialized memory by
//      design.
//
//   2. Not requiring a thread exit handler/implementation, keeping the
//      thread-local as purely a collection of references/data that do not
//      require cleanup.
//
// We're doing this to avoid using a `thread_local` object that has a
// non-trivial destructor, because the C++ runtime might call std::malloc(...)
// to register calls to destructors. Deadlocks may arise when, for example, an
// externally provided malloc implementation is XRay instrumented, and
// initializing the thread-locals involves calling into malloc. A malloc
// implementation that does global synchronization might be holding a lock for a
// critical section, calling a function that might be XRay instrumented (and
// thus in turn calling into malloc by virtue of registration of the
// thread_local's destructor).
static ThreadLocalData &getThreadLocalData() {
  static_assert(alignof(ThreadLocalData) >= 64,
                "ThreadLocalData must be cache line aligned.");
  thread_local ThreadLocalData TLD;
  thread_local bool UNUSED ThreadOnce = [] {
    pthread_setspecific(Key, &TLD);
    return false;
  }();
  return TLD;
}

//-----------------------------------------------------------------------------|
// The rest of the file is implementation.                                     |
//-----------------------------------------------------------------------------|
// Functions are implemented in the header for inlining since we don't want    |
// to grow the stack when we've hijacked the binary for logging.               |
//-----------------------------------------------------------------------------|

namespace {

class RecursionGuard {
  volatile bool &Running;
  const bool Valid;

public:
  explicit RecursionGuard(volatile bool &R) : Running(R), Valid(!R) {
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

} // namespace

static void writeNewBufferPreamble(pid_t Tid,
                                   timespec TS) XRAY_NEVER_INSTRUMENT {
  static constexpr int InitRecordsCount = 2;
  auto &TLD = getThreadLocalData();
  MetadataRecord Metadata[InitRecordsCount];
  {
    // Write out a MetadataRecord to signify that this is the start of a new
    // buffer, associated with a particular thread, with a new CPU.  For the
    // data, we have 15 bytes to squeeze as much information as we can.  At this
    // point we only write down the following bytes:
    //   - Thread ID (pid_t, 4 bytes)
    auto &NewBuffer = Metadata[0];
    NewBuffer.Type = uint8_t(RecordType::Metadata);
    NewBuffer.RecordKind = uint8_t(MetadataRecord::RecordKinds::NewBuffer);
    std::memcpy(&NewBuffer.Data, &Tid, sizeof(pid_t));
  }

  // Also write the WalltimeMarker record.
  {
    static_assert(sizeof(time_t) <= 8, "time_t needs to be at most 8 bytes");
    auto &WalltimeMarker = Metadata[1];
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

  TLD.NumConsecutiveFnEnters = 0;
  TLD.NumTailCalls = 0;
  if (TLD.BQ == nullptr || TLD.BQ->finalizing())
    return;
  std::memcpy(TLD.RecordPtr, Metadata, sizeof(Metadata));
  TLD.RecordPtr += sizeof(Metadata);
  // Since we write out the extents as the first metadata record of the
  // buffer, we need to write out the extents including the extents record.
  __sanitizer::atomic_store(&TLD.Buffer.Extents->Size, sizeof(Metadata),
                            __sanitizer::memory_order_release);
}

inline void setupNewBuffer(int (*wall_clock_reader)(
    clockid_t, struct timespec *)) XRAY_NEVER_INSTRUMENT {
  auto &TLD = getThreadLocalData();
  auto &B = TLD.Buffer;
  TLD.RecordPtr = static_cast<char *>(B.Buffer);
  pid_t Tid = syscall(SYS_gettid);
  timespec TS{0, 0};
  // This is typically clock_gettime, but callers have injection ability.
  wall_clock_reader(CLOCK_MONOTONIC, &TS);
  writeNewBufferPreamble(Tid, TS);
  TLD.NumConsecutiveFnEnters = 0;
  TLD.NumTailCalls = 0;
}

static void incrementExtents(size_t Add) {
  auto &TLD = getThreadLocalData();
  __sanitizer::atomic_fetch_add(&TLD.Buffer.Extents->Size, Add,
                                __sanitizer::memory_order_acq_rel);
}

static void decrementExtents(size_t Subtract) {
  auto &TLD = getThreadLocalData();
  __sanitizer::atomic_fetch_sub(&TLD.Buffer.Extents->Size, Subtract,
                                __sanitizer::memory_order_acq_rel);
}

inline void writeNewCPUIdMetadata(uint16_t CPU,
                                  uint64_t TSC) XRAY_NEVER_INSTRUMENT {
  auto &TLD = getThreadLocalData();
  MetadataRecord NewCPUId;
  NewCPUId.Type = uint8_t(RecordType::Metadata);
  NewCPUId.RecordKind = uint8_t(MetadataRecord::RecordKinds::NewCPUId);

  // The data for the New CPU will contain the following bytes:
  //   - CPU ID (uint16_t, 2 bytes)
  //   - Full TSC (uint64_t, 8 bytes)
  // Total = 10 bytes.
  std::memcpy(&NewCPUId.Data, &CPU, sizeof(CPU));
  std::memcpy(&NewCPUId.Data[sizeof(CPU)], &TSC, sizeof(TSC));
  std::memcpy(TLD.RecordPtr, &NewCPUId, sizeof(MetadataRecord));
  TLD.RecordPtr += sizeof(MetadataRecord);
  TLD.NumConsecutiveFnEnters = 0;
  TLD.NumTailCalls = 0;
  incrementExtents(sizeof(MetadataRecord));
}

inline void writeTSCWrapMetadata(uint64_t TSC) XRAY_NEVER_INSTRUMENT {
  auto &TLD = getThreadLocalData();
  MetadataRecord TSCWrap;
  TSCWrap.Type = uint8_t(RecordType::Metadata);
  TSCWrap.RecordKind = uint8_t(MetadataRecord::RecordKinds::TSCWrap);

  // The data for the TSCWrap record contains the following bytes:
  //   - Full TSC (uint64_t, 8 bytes)
  // Total = 8 bytes.
  std::memcpy(&TSCWrap.Data, &TSC, sizeof(TSC));
  std::memcpy(TLD.RecordPtr, &TSCWrap, sizeof(MetadataRecord));
  TLD.RecordPtr += sizeof(MetadataRecord);
  TLD.NumConsecutiveFnEnters = 0;
  TLD.NumTailCalls = 0;
  incrementExtents(sizeof(MetadataRecord));
}

// Call Argument metadata records store the arguments to a function in the
// order of their appearance; holes are not supported by the buffer format.
static inline void writeCallArgumentMetadata(uint64_t A) XRAY_NEVER_INSTRUMENT {
  auto &TLD = getThreadLocalData();
  MetadataRecord CallArg;
  CallArg.Type = uint8_t(RecordType::Metadata);
  CallArg.RecordKind = uint8_t(MetadataRecord::RecordKinds::CallArgument);

  std::memcpy(CallArg.Data, &A, sizeof(A));
  std::memcpy(TLD.RecordPtr, &CallArg, sizeof(MetadataRecord));
  TLD.RecordPtr += sizeof(MetadataRecord);
  incrementExtents(sizeof(MetadataRecord));
}

static inline void
writeFunctionRecord(int FuncId, uint32_t TSCDelta,
                    XRayEntryType EntryType) XRAY_NEVER_INSTRUMENT {
  FunctionRecord FuncRecord;
  FuncRecord.Type = uint8_t(RecordType::Function);
  // Only take 28 bits of the function id.
  FuncRecord.FuncId = FuncId & ~(0x0F << 28);
  FuncRecord.TSCDelta = TSCDelta;

  auto &TLD = getThreadLocalData();
  switch (EntryType) {
  case XRayEntryType::ENTRY:
    ++TLD.NumConsecutiveFnEnters;
    FuncRecord.RecordKind = uint8_t(FunctionRecord::RecordKinds::FunctionEnter);
    break;
  case XRayEntryType::LOG_ARGS_ENTRY:
    // We should not rewind functions with logged args.
    TLD.NumConsecutiveFnEnters = 0;
    TLD.NumTailCalls = 0;
    FuncRecord.RecordKind = uint8_t(FunctionRecord::RecordKinds::FunctionEnter);
    break;
  case XRayEntryType::EXIT:
    // If we've decided to log the function exit, we will never erase the log
    // before it.
    TLD.NumConsecutiveFnEnters = 0;
    TLD.NumTailCalls = 0;
    FuncRecord.RecordKind = uint8_t(FunctionRecord::RecordKinds::FunctionExit);
    break;
  case XRayEntryType::TAIL:
    // If we just entered the function we're tail exiting from or erased every
    // invocation since then, this function entry tail pair is a candidate to
    // be erased when the child function exits.
    if (TLD.NumConsecutiveFnEnters > 0) {
      ++TLD.NumTailCalls;
      TLD.NumConsecutiveFnEnters = 0;
    } else {
      // We will never be able to erase this tail call since we have logged
      // something in between the function entry and tail exit.
      TLD.NumTailCalls = 0;
      TLD.NumConsecutiveFnEnters = 0;
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

  std::memcpy(TLD.RecordPtr, &FuncRecord, sizeof(FunctionRecord));
  TLD.RecordPtr += sizeof(FunctionRecord);
  incrementExtents(sizeof(FunctionRecord));
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
  auto &TLD = getThreadLocalData();
  TLD.RecordPtr -= FunctionRecSize;
  decrementExtents(FunctionRecSize);
  FunctionRecord FuncRecord;
  std::memcpy(&FuncRecord, TLD.RecordPtr, FunctionRecSize);
  assert(FuncRecord.RecordKind ==
             uint8_t(FunctionRecord::RecordKinds::FunctionEnter) &&
         "Expected to find function entry recording when rewinding.");
  assert(FuncRecord.FuncId == (FuncId & ~(0x0F << 28)) &&
         "Expected matching function id when rewinding Exit");
  --TLD.NumConsecutiveFnEnters;
  LastTSC -= FuncRecord.TSCDelta;

  // We unwound one call. Update the state and return without writing a log.
  if (TLD.NumConsecutiveFnEnters != 0) {
    LastFunctionEntryTSC -= FuncRecord.TSCDelta;
    return;
  }

  // Otherwise we've rewound the stack of all function entries, we might be
  // able to rewind further by erasing tail call functions that are being
  // exited from via this exit.
  LastFunctionEntryTSC = 0;
  auto RewindingTSC = LastTSC;
  auto RewindingRecordPtr = TLD.RecordPtr - FunctionRecSize;
  while (TLD.NumTailCalls > 0) {
    // Rewind the TSC back over the TAIL EXIT record.
    FunctionRecord ExpectedTailExit;
    std::memcpy(&ExpectedTailExit, RewindingRecordPtr, FunctionRecSize);

    assert(ExpectedTailExit.RecordKind ==
               uint8_t(FunctionRecord::RecordKinds::FunctionTailExit) &&
           "Expected to find tail exit when rewinding.");
    RewindingRecordPtr -= FunctionRecSize;
    RewindingTSC -= ExpectedTailExit.TSCDelta;
    FunctionRecord ExpectedFunctionEntry;
    std::memcpy(&ExpectedFunctionEntry, RewindingRecordPtr, FunctionRecSize);
    assert(ExpectedFunctionEntry.RecordKind ==
               uint8_t(FunctionRecord::RecordKinds::FunctionEnter) &&
           "Expected to find function entry when rewinding tail call.");
    assert(ExpectedFunctionEntry.FuncId == ExpectedTailExit.FuncId &&
           "Expected funcids to match when rewinding tail call.");

    // This tail call exceeded the threshold duration. It will not be erased.
    if ((TSC - RewindingTSC) >= thresholdTicks()) {
      TLD.NumTailCalls = 0;
      return;
    }

    // We can erase a tail exit pair that we're exiting through since
    // its duration is under threshold.
    --TLD.NumTailCalls;
    RewindingRecordPtr -= FunctionRecSize;
    RewindingTSC -= ExpectedFunctionEntry.TSCDelta;
    TLD.RecordPtr -= 2 * FunctionRecSize;
    LastTSC = RewindingTSC;
    decrementExtents(2 * FunctionRecSize);
  }
}

inline bool releaseThreadLocalBuffer(BufferQueue &BQArg) {
  auto &TLD = getThreadLocalData();
  auto EC = BQArg.releaseBuffer(TLD.Buffer);
  if (EC != BufferQueue::ErrorCode::Ok) {
    Report("Failed to release buffer at %p; error=%s\n", TLD.Buffer.Buffer,
           BufferQueue::getErrorString(EC));
    return false;
  }
  return true;
}

inline bool prepareBuffer(uint64_t TSC, unsigned char CPU,
                          int (*wall_clock_reader)(clockid_t,
                                                   struct timespec *),
                          size_t MaxSize) XRAY_NEVER_INSTRUMENT {
  auto &TLD = getThreadLocalData();
  char *BufferStart = static_cast<char *>(TLD.Buffer.Buffer);
  if ((TLD.RecordPtr + MaxSize) > (BufferStart + TLD.Buffer.Size)) {
    if (!releaseThreadLocalBuffer(*TLD.BQ))
      return false;
    auto EC = TLD.BQ->getBuffer(TLD.Buffer);
    if (EC != BufferQueue::ErrorCode::Ok) {
      Report("Failed to acquire a buffer; error=%s\n",
             BufferQueue::getErrorString(EC));
      return false;
    }
    setupNewBuffer(wall_clock_reader);

    // Always write the CPU metadata as the first record in the buffer.
    writeNewCPUIdMetadata(CPU, TSC);
  }
  return true;
}

inline bool
isLogInitializedAndReady(BufferQueue *LBQ, uint64_t TSC, unsigned char CPU,
                         int (*wall_clock_reader)(clockid_t, struct timespec *))
    XRAY_NEVER_INSTRUMENT {
  // Bail out right away if logging is not initialized yet.
  // We should take the opportunity to release the buffer though.
  auto Status = __sanitizer::atomic_load(&LoggingStatus,
                                         __sanitizer::memory_order_acquire);
  auto &TLD = getThreadLocalData();
  if (Status != XRayLogInitStatus::XRAY_LOG_INITIALIZED) {
    if (TLD.RecordPtr != nullptr &&
        (Status == XRayLogInitStatus::XRAY_LOG_FINALIZING ||
         Status == XRayLogInitStatus::XRAY_LOG_FINALIZED)) {
      if (!releaseThreadLocalBuffer(*LBQ))
        return false;
      TLD.RecordPtr = nullptr;
      return false;
    }
    return false;
  }

  if (__sanitizer::atomic_load(&LoggingStatus,
                               __sanitizer::memory_order_acquire) !=
          XRayLogInitStatus::XRAY_LOG_INITIALIZED ||
      LBQ->finalizing()) {
    if (!releaseThreadLocalBuffer(*LBQ))
      return false;
    TLD.RecordPtr = nullptr;
  }

  if (TLD.Buffer.Buffer == nullptr) {
    auto EC = LBQ->getBuffer(TLD.Buffer);
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

    // Always write the CPU metadata as the first record in the buffer.
    writeNewCPUIdMetadata(CPU, TSC);
  }

  if (TLD.CurrentCPU == std::numeric_limits<uint16_t>::max()) {
    // This means this is the first CPU this thread has ever run on. We set
    // the current CPU and record this as the first TSC we've seen.
    TLD.CurrentCPU = CPU;
    writeNewCPUIdMetadata(CPU, TSC);
  }

  return true;
} // namespace __xray_fdr_internal

// Compute the TSC difference between the time of measurement and the previous
// event. There are a few interesting situations we need to account for:
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
inline uint32_t writeCurrentCPUTSC(ThreadLocalData &TLD, uint64_t TSC,
                                   uint8_t CPU) {
  if (CPU != TLD.CurrentCPU) {
    // We've moved to a new CPU.
    writeNewCPUIdMetadata(CPU, TSC);
    return 0;
  }
  // If the delta is greater than the range for a uint32_t, then we write out
  // the TSC wrap metadata entry with the full TSC, and the TSC for the
  // function record be 0.
  uint64_t Delta = TSC - TLD.LastTSC;
  if (Delta <= std::numeric_limits<uint32_t>::max())
    return Delta;

  writeTSCWrapMetadata(TSC);
  return 0;
}

inline void endBufferIfFull() XRAY_NEVER_INSTRUMENT {
  auto &TLD = getThreadLocalData();
  auto BufferStart = static_cast<char *>(TLD.Buffer.Buffer);
  if ((TLD.RecordPtr + MetadataRecSize) - BufferStart <=
      ptrdiff_t{MetadataRecSize}) {
    if (!releaseThreadLocalBuffer(*TLD.BQ))
      return;
    TLD.RecordPtr = nullptr;
  }
}

thread_local volatile bool Running = false;

/// Here's where the meat of the processing happens. The writer captures
/// function entry, exit and tail exit points with a time and will create
/// TSCWrap, NewCPUId and Function records as necessary. The writer might
/// walk backward through its buffer and erase trivial functions to avoid
/// polluting the log and may use the buffer queue to obtain or release a
/// buffer.
inline void processFunctionHook(int32_t FuncId, XRayEntryType Entry,
                                uint64_t TSC, unsigned char CPU, uint64_t Arg1,
                                int (*wall_clock_reader)(clockid_t,
                                                         struct timespec *),
                                BufferQueue *BQ) XRAY_NEVER_INSTRUMENT {
  // Prevent signal handler recursion, so in case we're already in a log writing
  // mode and the signal handler comes in (and is also instrumented) then we
  // don't want to be clobbering potentially partial writes already happening in
  // the thread. We use a simple thread_local latch to only allow one on-going
  // handleArg0 to happen at any given time.
  RecursionGuard Guard{Running};
  if (!Guard) {
    assert(Running == true && "RecursionGuard is buggy!");
    return;
  }

  auto &TLD = getThreadLocalData();

  // In case the reference has been cleaned up before, we make sure we
  // initialize it to the provided BufferQueue.
  if (TLD.BQ == nullptr)
    TLD.BQ = BQ;

  if (!isLogInitializedAndReady(TLD.BQ, TSC, CPU, wall_clock_reader))
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
  //   - The most number of bytes we will ever write is 8 + 16 + 16 = 40.
  //     This is computed by:
  //
  //       MaxSize = sizeof(FunctionRecord) + 2 * sizeof(MetadataRecord)
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
  //       4. The second MetadataRecord is the optional function call argument.
  //
  // So the math we need to do is to determine whether writing 40 bytes past the
  // current pointer exceeds the buffer's maximum size. If we don't have enough
  // space to write 40 bytes in the buffer, we need get a new Buffer, set it up
  // properly before doing any further writing.
  size_t MaxSize = FunctionRecSize + 2 * MetadataRecSize;
  if (!prepareBuffer(TSC, CPU, wall_clock_reader, MaxSize)) {
    TLD.BQ = nullptr;
    return;
  }

  // By this point, we are now ready to write up to 40 bytes (explained above).
  assert((TLD.RecordPtr + MaxSize) - static_cast<char *>(TLD.Buffer.Buffer) >=
             static_cast<ptrdiff_t>(MetadataRecSize) &&
         "Misconfigured BufferQueue provided; Buffer size not large enough.");

  auto RecordTSCDelta = writeCurrentCPUTSC(TLD, TSC, CPU);
  TLD.LastTSC = TSC;
  TLD.CurrentCPU = CPU;
  switch (Entry) {
  case XRayEntryType::ENTRY:
  case XRayEntryType::LOG_ARGS_ENTRY:
    // Update the thread local state for the next invocation.
    TLD.LastFunctionEntryTSC = TSC;
    break;
  case XRayEntryType::TAIL:
  case XRayEntryType::EXIT:
    // Break out and write the exit record if we can't erase any functions.
    if (TLD.NumConsecutiveFnEnters == 0 ||
        (TSC - TLD.LastFunctionEntryTSC) >= thresholdTicks())
      break;
    rewindRecentCall(TSC, TLD.LastTSC, TLD.LastFunctionEntryTSC, FuncId);
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

  writeFunctionRecord(FuncId, RecordTSCDelta, Entry);
  if (Entry == XRayEntryType::LOG_ARGS_ENTRY)
    writeCallArgumentMetadata(Arg1);

  // If we've exhausted the buffer by this time, we then release the buffer to
  // make sure that other threads may start using this buffer.
  endBufferIfFull();
}

} // namespace __xray_fdr_internal
} // namespace __xray

#endif // XRAY_XRAY_FDR_LOGGING_IMPL_H
