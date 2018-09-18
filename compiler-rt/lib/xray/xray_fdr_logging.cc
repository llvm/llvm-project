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
#include <cassert>
#include <errno.h>
#include <limits>
#include <memory>
#include <pthread.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#include "sanitizer_common/sanitizer_allocator_internal.h"
#include "sanitizer_common/sanitizer_atomic.h"
#include "sanitizer_common/sanitizer_common.h"
#include "xray/xray_interface.h"
#include "xray/xray_records.h"
#include "xray_allocator.h"
#include "xray_buffer_queue.h"
#include "xray_defs.h"
#include "xray_fdr_flags.h"
#include "xray_flags.h"
#include "xray_recursion_guard.h"
#include "xray_tsc.h"
#include "xray_utils.h"

namespace __xray {

atomic_sint32_t LoggingStatus = {XRayLogInitStatus::XRAY_LOG_UNINITIALIZED};

// Group together thread-local-data in a struct, then hide it behind a function
// call so that it can be initialized on first use instead of as a global. We
// force the alignment to 64-bytes for x86 cache line alignment, as this
// structure is used in the hot path of implementation.
struct alignas(64) ThreadLocalData {
  BufferQueue::Buffer Buffer{};
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

// Global BufferQueue.
static std::aligned_storage<sizeof(BufferQueue)>::type BufferQueueStorage;
static BufferQueue *BQ = nullptr;

static atomic_sint32_t LogFlushStatus = {
    XRayLogFlushStatus::XRAY_LOG_NOT_FLUSHING};

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
static_assert(alignof(ThreadLocalData) >= 64,
              "ThreadLocalData must be cache line aligned.");
static ThreadLocalData &getThreadLocalData() {
  thread_local typename std::aligned_storage<
      sizeof(ThreadLocalData), alignof(ThreadLocalData)>::type TLDStorage{};

  if (pthread_getspecific(Key) == NULL) {
    new (reinterpret_cast<ThreadLocalData *>(&TLDStorage)) ThreadLocalData{};
    pthread_setspecific(Key, &TLDStorage);
  }

  return *reinterpret_cast<ThreadLocalData *>(&TLDStorage);
}

static void writeNewBufferPreamble(tid_t Tid, timespec TS,
                                   pid_t Pid) XRAY_NEVER_INSTRUMENT {
  static constexpr int InitRecordsCount = 3;
  auto &TLD = getThreadLocalData();
  MetadataRecord Metadata[InitRecordsCount];
  {
    // Write out a MetadataRecord to signify that this is the start of a new
    // buffer, associated with a particular thread, with a new CPU.  For the
    // data, we have 15 bytes to squeeze as much information as we can.  At this
    // point we only write down the following bytes:
    //   - Thread ID (tid_t, cast to 4 bytes type due to Darwin being 8 bytes)
    auto &NewBuffer = Metadata[0];
    NewBuffer.Type = uint8_t(RecordType::Metadata);
    NewBuffer.RecordKind = uint8_t(MetadataRecord::RecordKinds::NewBuffer);
    int32_t tid = static_cast<int32_t>(Tid);
    internal_memcpy(&NewBuffer.Data, &tid, sizeof(tid));
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
    internal_memcpy(WalltimeMarker.Data, &Seconds, sizeof(Seconds));
    internal_memcpy(WalltimeMarker.Data + sizeof(Seconds), &Micros,
                    sizeof(Micros));
  }

  // Also write the Pid record.
  {
    // Write out a MetadataRecord that contains the current pid
    auto &PidMetadata = Metadata[2];
    PidMetadata.Type = uint8_t(RecordType::Metadata);
    PidMetadata.RecordKind = uint8_t(MetadataRecord::RecordKinds::Pid);
    int32_t pid = static_cast<int32_t>(Pid);
    internal_memcpy(&PidMetadata.Data, &pid, sizeof(pid));
  }

  TLD.NumConsecutiveFnEnters = 0;
  TLD.NumTailCalls = 0;
  if (TLD.BQ == nullptr || TLD.BQ->finalizing())
    return;
  internal_memcpy(TLD.RecordPtr, Metadata,
                  sizeof(MetadataRecord) * InitRecordsCount);
  TLD.RecordPtr += sizeof(MetadataRecord) * InitRecordsCount;
  atomic_store(&TLD.Buffer.Extents, sizeof(MetadataRecord) * InitRecordsCount,
               memory_order_release);
}

static void setupNewBuffer(int (*wall_clock_reader)(
    clockid_t, struct timespec *)) XRAY_NEVER_INSTRUMENT {
  auto &TLD = getThreadLocalData();
  auto &B = TLD.Buffer;
  TLD.RecordPtr = static_cast<char *>(B.Data);
  tid_t Tid = GetTid();
  timespec TS{0, 0};
  pid_t Pid = internal_getpid();
  // This is typically clock_gettime, but callers have injection ability.
  wall_clock_reader(CLOCK_MONOTONIC, &TS);
  writeNewBufferPreamble(Tid, TS, Pid);
  TLD.NumConsecutiveFnEnters = 0;
  TLD.NumTailCalls = 0;
}

static void incrementExtents(size_t Add) {
  auto &TLD = getThreadLocalData();
  atomic_fetch_add(&TLD.Buffer.Extents, Add, memory_order_acq_rel);
}

static void decrementExtents(size_t Subtract) {
  auto &TLD = getThreadLocalData();
  atomic_fetch_sub(&TLD.Buffer.Extents, Subtract, memory_order_acq_rel);
}

static void writeNewCPUIdMetadata(uint16_t CPU,
                                  uint64_t TSC) XRAY_NEVER_INSTRUMENT {
  auto &TLD = getThreadLocalData();
  MetadataRecord NewCPUId;
  NewCPUId.Type = uint8_t(RecordType::Metadata);
  NewCPUId.RecordKind = uint8_t(MetadataRecord::RecordKinds::NewCPUId);

  // The data for the New CPU will contain the following bytes:
  //   - CPU ID (uint16_t, 2 bytes)
  //   - Full TSC (uint64_t, 8 bytes)
  // Total = 10 bytes.
  internal_memcpy(&NewCPUId.Data, &CPU, sizeof(CPU));
  internal_memcpy(&NewCPUId.Data[sizeof(CPU)], &TSC, sizeof(TSC));
  internal_memcpy(TLD.RecordPtr, &NewCPUId, sizeof(MetadataRecord));
  TLD.RecordPtr += sizeof(MetadataRecord);
  TLD.NumConsecutiveFnEnters = 0;
  TLD.NumTailCalls = 0;
  incrementExtents(sizeof(MetadataRecord));
}

static void writeTSCWrapMetadata(uint64_t TSC) XRAY_NEVER_INSTRUMENT {
  auto &TLD = getThreadLocalData();
  MetadataRecord TSCWrap;
  TSCWrap.Type = uint8_t(RecordType::Metadata);
  TSCWrap.RecordKind = uint8_t(MetadataRecord::RecordKinds::TSCWrap);

  // The data for the TSCWrap record contains the following bytes:
  //   - Full TSC (uint64_t, 8 bytes)
  // Total = 8 bytes.
  internal_memcpy(&TSCWrap.Data, &TSC, sizeof(TSC));
  internal_memcpy(TLD.RecordPtr, &TSCWrap, sizeof(MetadataRecord));
  TLD.RecordPtr += sizeof(MetadataRecord);
  TLD.NumConsecutiveFnEnters = 0;
  TLD.NumTailCalls = 0;
  incrementExtents(sizeof(MetadataRecord));
}

// Call Argument metadata records store the arguments to a function in the
// order of their appearance; holes are not supported by the buffer format.
static void writeCallArgumentMetadata(uint64_t A) XRAY_NEVER_INSTRUMENT {
  auto &TLD = getThreadLocalData();
  MetadataRecord CallArg;
  CallArg.Type = uint8_t(RecordType::Metadata);
  CallArg.RecordKind = uint8_t(MetadataRecord::RecordKinds::CallArgument);

  internal_memcpy(CallArg.Data, &A, sizeof(A));
  internal_memcpy(TLD.RecordPtr, &CallArg, sizeof(MetadataRecord));
  TLD.RecordPtr += sizeof(MetadataRecord);
  incrementExtents(sizeof(MetadataRecord));
}

static void writeFunctionRecord(int32_t FuncId, uint32_t TSCDelta,
                                XRayEntryType EntryType) XRAY_NEVER_INSTRUMENT {
  constexpr int32_t MaxFuncId = (1 << 29) - 1;
  if (UNLIKELY(FuncId > MaxFuncId)) {
    if (Verbosity())
      Report("Warning: Function ID '%d' > max function id: '%d'", FuncId,
             MaxFuncId);
    return;
  }

  FunctionRecord FuncRecord;
  FuncRecord.Type = uint8_t(RecordType::Function);

  // Only take 28 bits of the function id.
  //
  // We need to be careful about the sign bit and the bitwise operations being
  // performed here. In effect, we want to truncate the value of the function id
  // to the first 28 bits. To do this properly, this means we need to mask the
  // function id with (2 ^ 28) - 1 == 0x0fffffff.
  //
  FuncRecord.FuncId = FuncId & MaxFuncId;
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
    static atomic_uint8_t ErrorLatch{0};
    if (!atomic_exchange(&ErrorLatch, 1, memory_order_acq_rel))
      Report("Internal error: patched an XRay custom event call as a function; "
             "func id = %d\n",
             FuncId);
    return;
  }
  case XRayEntryType::TYPED_EVENT: {
    static atomic_uint8_t ErrorLatch{0};
    if (!atomic_exchange(&ErrorLatch, 1, memory_order_acq_rel))
      Report("Internal error: patched an XRay typed event call as a function; "
             "func id = %d\n",
             FuncId);
    return;
  }
  }

  internal_memcpy(TLD.RecordPtr, &FuncRecord, sizeof(FunctionRecord));
  TLD.RecordPtr += sizeof(FunctionRecord);
  incrementExtents(sizeof(FunctionRecord));
}

static atomic_uint64_t TicksPerSec{0};
static atomic_uint64_t ThresholdTicks{0};

// Re-point the thread local pointer into this thread's Buffer before the recent
// "Function Entry" record and any "Tail Call Exit" records after that.
static void rewindRecentCall(uint64_t TSC, uint64_t &LastTSC,
                             uint64_t &LastFunctionEntryTSC,
                             int32_t FuncId) XRAY_NEVER_INSTRUMENT {
  auto &TLD = getThreadLocalData();
  TLD.RecordPtr -= FunctionRecSize;
  decrementExtents(FunctionRecSize);
  FunctionRecord FuncRecord;
  internal_memcpy(&FuncRecord, TLD.RecordPtr, FunctionRecSize);
  DCHECK(FuncRecord.RecordKind ==
             uint8_t(FunctionRecord::RecordKinds::FunctionEnter) &&
         "Expected to find function entry recording when rewinding.");
  DCHECK(FuncRecord.FuncId == (FuncId & ~(0x0F << 28)) &&
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
    internal_memcpy(&ExpectedTailExit, RewindingRecordPtr, FunctionRecSize);

    DCHECK(ExpectedTailExit.RecordKind ==
               uint8_t(FunctionRecord::RecordKinds::FunctionTailExit) &&
           "Expected to find tail exit when rewinding.");
    RewindingRecordPtr -= FunctionRecSize;
    RewindingTSC -= ExpectedTailExit.TSCDelta;
    FunctionRecord ExpectedFunctionEntry;
    internal_memcpy(&ExpectedFunctionEntry, RewindingRecordPtr,
                    FunctionRecSize);
    DCHECK(ExpectedFunctionEntry.RecordKind ==
               uint8_t(FunctionRecord::RecordKinds::FunctionEnter) &&
           "Expected to find function entry when rewinding tail call.");
    DCHECK(ExpectedFunctionEntry.FuncId == ExpectedTailExit.FuncId &&
           "Expected funcids to match when rewinding tail call.");

    // This tail call exceeded the threshold duration. It will not be erased.
    if ((TSC - RewindingTSC) >= atomic_load_relaxed(&ThresholdTicks)) {
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

static bool releaseThreadLocalBuffer(BufferQueue &BQArg) {
  auto &TLD = getThreadLocalData();
  auto EC = BQArg.releaseBuffer(TLD.Buffer);
  if (EC != BufferQueue::ErrorCode::Ok) {
    Report("Failed to release buffer at %p; error=%s\n", TLD.Buffer.Data,
           BufferQueue::getErrorString(EC));
    return false;
  }
  return true;
}

static bool prepareBuffer(uint64_t TSC, unsigned char CPU,
                          int (*wall_clock_reader)(clockid_t,
                                                   struct timespec *),
                          size_t MaxSize) XRAY_NEVER_INSTRUMENT {
  auto &TLD = getThreadLocalData();
  char *BufferStart = static_cast<char *>(TLD.Buffer.Data);
  if ((TLD.RecordPtr + MaxSize) > (BufferStart + TLD.Buffer.Size)) {
    if (!releaseThreadLocalBuffer(*TLD.BQ))
      return false;
    auto EC = TLD.BQ->getBuffer(TLD.Buffer);
    if (EC != BufferQueue::ErrorCode::Ok) {
      Report("Failed to prepare a buffer; error = '%s'\n",
             BufferQueue::getErrorString(EC));
      return false;
    }
    setupNewBuffer(wall_clock_reader);

    // Always write the CPU metadata as the first record in the buffer.
    writeNewCPUIdMetadata(CPU, TSC);
  }
  return true;
}

static bool
isLogInitializedAndReady(BufferQueue *LBQ, uint64_t TSC, unsigned char CPU,
                         int (*wall_clock_reader)(clockid_t, struct timespec *))
    XRAY_NEVER_INSTRUMENT {
  // Bail out right away if logging is not initialized yet.
  // We should take the opportunity to release the buffer though.
  auto Status = atomic_load(&LoggingStatus, memory_order_acquire);
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

  if (atomic_load(&LoggingStatus, memory_order_acquire) !=
          XRayLogInitStatus::XRAY_LOG_INITIALIZED ||
      LBQ->finalizing()) {
    if (!releaseThreadLocalBuffer(*LBQ))
      return false;
    TLD.RecordPtr = nullptr;
  }

  if (TLD.Buffer.Data == nullptr) {
    auto EC = LBQ->getBuffer(TLD.Buffer);
    if (EC != BufferQueue::ErrorCode::Ok) {
      auto LS = atomic_load(&LoggingStatus, memory_order_acquire);
      if (LS != XRayLogInitStatus::XRAY_LOG_FINALIZING &&
          LS != XRayLogInitStatus::XRAY_LOG_FINALIZED)
        Report("Failed to acquire a buffer; error = '%s'\n",
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
}

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
static uint32_t writeCurrentCPUTSC(ThreadLocalData &TLD, uint64_t TSC,
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

static void endBufferIfFull() XRAY_NEVER_INSTRUMENT {
  auto &TLD = getThreadLocalData();
  auto BufferStart = static_cast<char *>(TLD.Buffer.Data);
  if ((TLD.RecordPtr + MetadataRecSize) - BufferStart <=
      ptrdiff_t{MetadataRecSize}) {
    if (!releaseThreadLocalBuffer(*TLD.BQ))
      return;
    TLD.RecordPtr = nullptr;
  }
}

thread_local atomic_uint8_t Running{0};

/// Here's where the meat of the processing happens. The writer captures
/// function entry, exit and tail exit points with a time and will create
/// TSCWrap, NewCPUId and Function records as necessary. The writer might
/// walk backward through its buffer and erase trivial functions to avoid
/// polluting the log and may use the buffer queue to obtain or release a
/// buffer.
static void processFunctionHook(int32_t FuncId, XRayEntryType Entry,
                                uint64_t TSC, unsigned char CPU, uint64_t Arg1,
                                int (*wall_clock_reader)(clockid_t,
                                                         struct timespec *))
    XRAY_NEVER_INSTRUMENT {
  __asm volatile("# LLVM-MCA-BEGIN processFunctionHook");
  // Prevent signal handler recursion, so in case we're already in a log writing
  // mode and the signal handler comes in (and is also instrumented) then we
  // don't want to be clobbering potentially partial writes already happening in
  // the thread. We use a simple thread_local latch to only allow one on-going
  // handleArg0 to happen at any given time.
  RecursionGuard Guard{Running};
  if (!Guard) {
    DCHECK(atomic_load_relaxed(&Running) && "RecursionGuard is buggy!");
    return;
  }

  auto &TLD = getThreadLocalData();

  if (TLD.BQ == nullptr && BQ != nullptr)
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
  //          FunctionRecord. This means we have: 1 MetadataRecord + 1 Function
  //          Record.
  //       2. When we learn that we've moved CPUs, we need to write a
  //          MetadataRecord to indicate a "cpu change", and thus write out the
  //          current TSC for that CPU before writing out the actual
  //          FunctionRecord. This means we have: 1 MetadataRecord + 1 Function
  //          Record.
  //       3. Given the previous two cases, in addition we can add at most one
  //          function argument record. This means we have: 2 MetadataRecord + 1
  //          Function Record.
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

  auto RecordTSCDelta = writeCurrentCPUTSC(TLD, TSC, CPU);
  TLD.LastTSC = TSC;
  TLD.CurrentCPU = CPU;
  switch (Entry) {
  case XRayEntryType::ENTRY:
    // Update the thread local state for the next invocation.
    TLD.LastFunctionEntryTSC = TSC;
    break;
  case XRayEntryType::LOG_ARGS_ENTRY:
    // Update the thread local state for the next invocation, but also prevent
    // rewinding when we have arguments logged.
    TLD.LastFunctionEntryTSC = TSC;
    TLD.NumConsecutiveFnEnters = 0;
    TLD.NumTailCalls = 0;
    break;
  case XRayEntryType::TAIL:
  case XRayEntryType::EXIT:
    // Break out and write the exit record if we can't erase any functions.
    if (TLD.NumConsecutiveFnEnters == 0 ||
        (TSC - TLD.LastFunctionEntryTSC) >=
            atomic_load_relaxed(&ThresholdTicks))
      break;
    rewindRecentCall(TSC, TLD.LastTSC, TLD.LastFunctionEntryTSC, FuncId);
    return; // without writing log.
  case XRayEntryType::CUSTOM_EVENT: {
    // This is a bug in patching, so we'll report it once and move on.
    static atomic_uint8_t ErrorLatch{0};
    if (!atomic_exchange(&ErrorLatch, 1, memory_order_acq_rel))
      Report("Internal error: patched an XRay custom event call as a function; "
             "func id = %d\n",
             FuncId);
    return;
  }
  case XRayEntryType::TYPED_EVENT: {
    static atomic_uint8_t ErrorLatch{0};
    if (!atomic_exchange(&ErrorLatch, 1, memory_order_acq_rel))
      Report("Internal error: patched an XRay typed event call as a function; "
             "func id = %d\n",
             FuncId);
    return;
  }
  }

  writeFunctionRecord(FuncId, RecordTSCDelta, Entry);
  if (Entry == XRayEntryType::LOG_ARGS_ENTRY)
    writeCallArgumentMetadata(Arg1);

  // If we've exhausted the buffer by this time, we then release the buffer to
  // make sure that other threads may start using this buffer.
  endBufferIfFull();
  __asm volatile("# LLVM-MCA-END");
}

static XRayFileHeader &fdrCommonHeaderInfo() {
  static std::aligned_storage<sizeof(XRayFileHeader)>::type HStorage;
  static pthread_once_t OnceInit = PTHREAD_ONCE_INIT;
  static bool TSCSupported = true;
  static uint64_t CycleFrequency = NanosecondsPerSecond;
  pthread_once(&OnceInit, +[] {
    XRayFileHeader &H = reinterpret_cast<XRayFileHeader &>(HStorage);
    // Version 2 of the log writes the extents of the buffer, instead of
    // relying on an end-of-buffer record.
    // Version 3 includes PID metadata record
    H.Version = 3;
    H.Type = FileTypes::FDR_LOG;

    // Test for required CPU features and cache the cycle frequency
    TSCSupported = probeRequiredCPUFeatures();
    if (TSCSupported)
      CycleFrequency = getTSCFrequency();
    H.CycleFrequency = CycleFrequency;

    // FIXME: Actually check whether we have 'constant_tsc' and
    // 'nonstop_tsc' before setting the values in the header.
    H.ConstantTSC = 1;
    H.NonstopTSC = 1;
  });
  return reinterpret_cast<XRayFileHeader &>(HStorage);
}

// This is the iterator implementation, which knows how to handle FDR-mode
// specific buffers. This is used as an implementation of the iterator function
// needed by __xray_set_buffer_iterator(...). It maintains a global state of the
// buffer iteration for the currently installed FDR mode buffers. In particular:
//
//   - If the argument represents the initial state of XRayBuffer ({nullptr, 0})
//     then the iterator returns the header information.
//   - If the argument represents the header information ({address of header
//     info, size of the header info}) then it returns the first FDR buffer's
//     address and extents.
//   - It will keep returning the next buffer and extents as there are more
//     buffers to process. When the input represents the last buffer, it will
//     return the initial state to signal completion ({nullptr, 0}).
//
// See xray/xray_log_interface.h for more details on the requirements for the
// implementations of __xray_set_buffer_iterator(...) and
// __xray_log_process_buffers(...).
XRayBuffer fdrIterator(const XRayBuffer B) {
  DCHECK(internal_strcmp(__xray_log_get_current_mode(), "xray-fdr") == 0);
  DCHECK(BQ->finalizing());

  if (BQ == nullptr || !BQ->finalizing()) {
    if (Verbosity())
      Report(
          "XRay FDR: Failed global buffer queue is null or not finalizing!\n");
    return {nullptr, 0};
  }

  // We use a global scratch-pad for the header information, which only gets
  // initialized the first time this function is called. We'll update one part
  // of this information with some relevant data (in particular the number of
  // buffers to expect).
  static std::aligned_storage<sizeof(XRayFileHeader)>::type HeaderStorage;
  static pthread_once_t HeaderOnce = PTHREAD_ONCE_INIT;
  pthread_once(&HeaderOnce, +[] {
    reinterpret_cast<XRayFileHeader &>(HeaderStorage) = fdrCommonHeaderInfo();
  });

  // We use a convenience alias for code referring to Header from here on out.
  auto &Header = reinterpret_cast<XRayFileHeader &>(HeaderStorage);
  if (B.Data == nullptr && B.Size == 0) {
    Header.FdrData = FdrAdditionalHeaderData{BQ->ConfiguredBufferSize()};
    return XRayBuffer{static_cast<void *>(&Header), sizeof(Header)};
  }

  static BufferQueue::const_iterator It{};
  static BufferQueue::const_iterator End{};
  static void *CurrentBuffer{nullptr};
  static size_t SerializedBufferSize = 0;
  if (B.Data == static_cast<void *>(&Header) && B.Size == sizeof(Header)) {
    // From this point on, we provide raw access to the raw buffer we're getting
    // from the BufferQueue. We're relying on the iterators from the current
    // Buffer queue.
    It = BQ->cbegin();
    End = BQ->cend();
  }

  if (CurrentBuffer != nullptr) {
    deallocateBuffer(CurrentBuffer, SerializedBufferSize);
    CurrentBuffer = nullptr;
  }

  if (It == End)
    return {nullptr, 0};

  // Set up the current buffer to contain the extents like we would when writing
  // out to disk. The difference here would be that we still write "empty"
  // buffers, or at least go through the iterators faithfully to let the
  // handlers see the empty buffers in the queue.
  auto BufferSize = atomic_load(&It->Extents, memory_order_acquire);
  SerializedBufferSize = BufferSize + sizeof(MetadataRecord);
  CurrentBuffer = allocateBuffer(SerializedBufferSize);
  if (CurrentBuffer == nullptr)
    return {nullptr, 0};

  // Write out the extents as a Metadata Record into the CurrentBuffer.
  MetadataRecord ExtentsRecord;
  ExtentsRecord.Type = uint8_t(RecordType::Metadata);
  ExtentsRecord.RecordKind =
      uint8_t(MetadataRecord::RecordKinds::BufferExtents);
  internal_memcpy(ExtentsRecord.Data, &BufferSize, sizeof(BufferSize));
  auto AfterExtents =
      static_cast<char *>(internal_memcpy(CurrentBuffer, &ExtentsRecord,
                                          sizeof(MetadataRecord))) +
      sizeof(MetadataRecord);
  internal_memcpy(AfterExtents, It->Data, BufferSize);

  XRayBuffer Result;
  Result.Data = CurrentBuffer;
  Result.Size = SerializedBufferSize;
  ++It;
  return Result;
}

// Must finalize before flushing.
XRayLogFlushStatus fdrLoggingFlush() XRAY_NEVER_INSTRUMENT {
  if (atomic_load(&LoggingStatus, memory_order_acquire) !=
      XRayLogInitStatus::XRAY_LOG_FINALIZED) {
    if (Verbosity())
      Report("Not flushing log, implementation is not finalized.\n");
    return XRayLogFlushStatus::XRAY_LOG_NOT_FLUSHING;
  }

  s32 Result = XRayLogFlushStatus::XRAY_LOG_NOT_FLUSHING;
  if (!atomic_compare_exchange_strong(&LogFlushStatus, &Result,
                                      XRayLogFlushStatus::XRAY_LOG_FLUSHING,
                                      memory_order_release)) {
    if (Verbosity())
      Report("Not flushing log, implementation is still finalizing.\n");
    return static_cast<XRayLogFlushStatus>(Result);
  }

  if (BQ == nullptr) {
    if (Verbosity())
      Report("Cannot flush when global buffer queue is null.\n");
    return XRayLogFlushStatus::XRAY_LOG_NOT_FLUSHING;
  }

  // We wait a number of milliseconds to allow threads to see that we've
  // finalised before attempting to flush the log.
  SleepForMillis(fdrFlags()->grace_period_ms);

  // At this point, we're going to uninstall the iterator implementation, before
  // we decide to do anything further with the global buffer queue.
  __xray_log_remove_buffer_iterator();

  // Once flushed, we should set the global status of the logging implementation
  // to "uninitialized" to allow for FDR-logging multiple runs.
  auto ResetToUnitialized = at_scope_exit([] {
    atomic_store(&LoggingStatus, XRayLogInitStatus::XRAY_LOG_UNINITIALIZED,
                 memory_order_release);
  });

  auto CleanupBuffers = at_scope_exit([] {
    if (BQ != nullptr) {
      auto &TLD = getThreadLocalData();
      if (TLD.RecordPtr != nullptr && TLD.BQ != nullptr)
        releaseThreadLocalBuffer(*TLD.BQ);
      BQ->~BufferQueue();
      BQ = nullptr;
    }
  });

  if (fdrFlags()->no_file_flush) {
    if (Verbosity())
      Report("XRay FDR: Not flushing to file, 'no_file_flush=true'.\n");

    atomic_store(&LogFlushStatus, XRayLogFlushStatus::XRAY_LOG_FLUSHED,
                 memory_order_release);
    return XRayLogFlushStatus::XRAY_LOG_FLUSHED;
  }

  // We write out the file in the following format:
  //
  //   1) We write down the XRay file header with version 1, type FDR_LOG.
  //   2) Then we use the 'apply' member of the BufferQueue that's live, to
  //      ensure that at this point in time we write down the buffers that have
  //      been released (and marked "used") -- we dump the full buffer for now
  //      (fixed-sized) and let the tools reading the buffers deal with the data
  //      afterwards.
  //
  int Fd = getLogFD();
  if (Fd == -1) {
    auto Result = XRayLogFlushStatus::XRAY_LOG_NOT_FLUSHING;
    atomic_store(&LogFlushStatus, Result, memory_order_release);
    return Result;
  }

  XRayFileHeader Header = fdrCommonHeaderInfo();
  Header.FdrData = FdrAdditionalHeaderData{BQ->ConfiguredBufferSize()};
  retryingWriteAll(Fd, reinterpret_cast<char *>(&Header),
                   reinterpret_cast<char *>(&Header) + sizeof(Header));

  // Release the current thread's buffer before we attempt to write out all the
  // buffers. This ensures that in case we had only a single thread going, that
  // we are able to capture the data nonetheless.
  auto &TLD = getThreadLocalData();
  if (TLD.RecordPtr != nullptr && TLD.BQ != nullptr)
    releaseThreadLocalBuffer(*TLD.BQ);

  BQ->apply([&](const BufferQueue::Buffer &B) {
    // Starting at version 2 of the FDR logging implementation, we only write
    // the records identified by the extents of the buffer. We use the Extents
    // from the Buffer and write that out as the first record in the buffer.  We
    // still use a Metadata record, but fill in the extents instead for the
    // data.
    MetadataRecord ExtentsRecord;
    auto BufferExtents = atomic_load(&B.Extents, memory_order_acquire);
    DCHECK(BufferExtents <= B.Size);
    ExtentsRecord.Type = uint8_t(RecordType::Metadata);
    ExtentsRecord.RecordKind =
        uint8_t(MetadataRecord::RecordKinds::BufferExtents);
    internal_memcpy(ExtentsRecord.Data, &BufferExtents, sizeof(BufferExtents));
    if (BufferExtents > 0) {
      retryingWriteAll(Fd, reinterpret_cast<char *>(&ExtentsRecord),
                       reinterpret_cast<char *>(&ExtentsRecord) +
                           sizeof(MetadataRecord));
      retryingWriteAll(Fd, reinterpret_cast<char *>(B.Data),
                       reinterpret_cast<char *>(B.Data) + BufferExtents);
    }
  });

  atomic_store(&LogFlushStatus, XRayLogFlushStatus::XRAY_LOG_FLUSHED,
               memory_order_release);
  return XRayLogFlushStatus::XRAY_LOG_FLUSHED;
}

XRayLogInitStatus fdrLoggingFinalize() XRAY_NEVER_INSTRUMENT {
  s32 CurrentStatus = XRayLogInitStatus::XRAY_LOG_INITIALIZED;
  if (!atomic_compare_exchange_strong(&LoggingStatus, &CurrentStatus,
                                      XRayLogInitStatus::XRAY_LOG_FINALIZING,
                                      memory_order_release)) {
    if (Verbosity())
      Report("Cannot finalize log, implementation not initialized.\n");
    return static_cast<XRayLogInitStatus>(CurrentStatus);
  }

  // Do special things to make the log finalize itself, and not allow any more
  // operations to be performed until re-initialized.
  if (BQ == nullptr) {
    if (Verbosity())
      Report("Attempting to finalize an uninitialized global buffer!\n");
  } else {
    BQ->finalize();
  }

  atomic_store(&LoggingStatus, XRayLogInitStatus::XRAY_LOG_FINALIZED,
               memory_order_release);
  return XRayLogInitStatus::XRAY_LOG_FINALIZED;
}

struct TSCAndCPU {
  uint64_t TSC = 0;
  unsigned char CPU = 0;
};

static TSCAndCPU getTimestamp() XRAY_NEVER_INSTRUMENT {
  // We want to get the TSC as early as possible, so that we can check whether
  // we've seen this CPU before. We also do it before we load anything else,
  // to allow for forward progress with the scheduling.
  TSCAndCPU Result;

  // Test once for required CPU features
  static pthread_once_t OnceProbe = PTHREAD_ONCE_INIT;
  static bool TSCSupported = true;
  pthread_once(&OnceProbe, +[] { TSCSupported = probeRequiredCPUFeatures(); });

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
  processFunctionHook(FuncId, Entry, TC.TSC, TC.CPU, 0, clock_gettime);
}

void fdrLoggingHandleArg1(int32_t FuncId, XRayEntryType Entry,
                          uint64_t Arg) XRAY_NEVER_INSTRUMENT {
  auto TC = getTimestamp();
  processFunctionHook(FuncId, Entry, TC.TSC, TC.CPU, Arg, clock_gettime);
}

void fdrLoggingHandleCustomEvent(void *Event,
                                 std::size_t EventSize) XRAY_NEVER_INSTRUMENT {
  auto TC = getTimestamp();
  auto &TSC = TC.TSC;
  auto &CPU = TC.CPU;
  RecursionGuard Guard{Running};
  if (!Guard)
    return;
  if (EventSize > std::numeric_limits<int32_t>::max()) {
    static pthread_once_t Once = PTHREAD_ONCE_INIT;
    pthread_once(&Once, +[] { Report("Event size too large.\n"); });
  }
  int32_t ReducedEventSize = static_cast<int32_t>(EventSize);
  auto &TLD = getThreadLocalData();
  if (!isLogInitializedAndReady(TLD.BQ, TSC, CPU, clock_gettime))
    return;

  // Here we need to prepare the log to handle:
  //   - The metadata record we're going to write. (16 bytes)
  //   - The additional data we're going to write. Currently, that's the size
  //   of the event we're going to dump into the log as free-form bytes.
  if (!prepareBuffer(TSC, CPU, clock_gettime,
                     MetadataRecSize + ReducedEventSize)) {
    TLD.BQ = nullptr;
    return;
  }

  // We need to reset the counts for the number of functions we're able to
  // rewind.
  TLD.NumConsecutiveFnEnters = 0;
  TLD.NumTailCalls = 0;

  // Write the custom event metadata record, which consists of the following
  // information:
  //   - 8 bytes (64-bits) for the full TSC when the event started.
  //   - 4 bytes (32-bits) for the length of the data.
  MetadataRecord CustomEvent;
  CustomEvent.Type = uint8_t(RecordType::Metadata);
  CustomEvent.RecordKind =
      uint8_t(MetadataRecord::RecordKinds::CustomEventMarker);
  constexpr auto TSCSize = sizeof(TC.TSC);
  internal_memcpy(&CustomEvent.Data, &ReducedEventSize, sizeof(int32_t));
  internal_memcpy(&CustomEvent.Data + sizeof(int32_t), &TSC, TSCSize);
  internal_memcpy(TLD.RecordPtr, &CustomEvent, sizeof(CustomEvent));
  TLD.RecordPtr += sizeof(CustomEvent);
  internal_memcpy(TLD.RecordPtr, Event, ReducedEventSize);
  TLD.RecordPtr += ReducedEventSize;
  incrementExtents(MetadataRecSize + ReducedEventSize);
  endBufferIfFull();
}

void fdrLoggingHandleTypedEvent(
    uint16_t EventType, const void *Event,
    std::size_t EventSize) noexcept XRAY_NEVER_INSTRUMENT {
  auto TC = getTimestamp();
  auto &TSC = TC.TSC;
  auto &CPU = TC.CPU;
  RecursionGuard Guard{Running};
  if (!Guard)
    return;
  if (EventSize > std::numeric_limits<int32_t>::max()) {
    static pthread_once_t Once = PTHREAD_ONCE_INIT;
    pthread_once(&Once, +[] { Report("Event size too large.\n"); });
  }
  int32_t ReducedEventSize = static_cast<int32_t>(EventSize);
  auto &TLD = getThreadLocalData();
  if (!isLogInitializedAndReady(TLD.BQ, TSC, CPU, clock_gettime))
    return;

  // Here we need to prepare the log to handle:
  //   - The metadata record we're going to write. (16 bytes)
  //   - The additional data we're going to write. Currently, that's the size
  //   of the event we're going to dump into the log as free-form bytes.
  if (!prepareBuffer(TSC, CPU, clock_gettime,
                     MetadataRecSize + ReducedEventSize)) {
    TLD.BQ = nullptr;
    return;
  }
  // Write the custom event metadata record, which consists of the following
  // information:
  //   - 8 bytes (64-bits) for the full TSC when the event started.
  //   - 4 bytes (32-bits) for the length of the data.
  //   - 2 bytes (16-bits) for the event type. 3 bytes remain since one of the
  //       bytes has the record type (Metadata Record) and kind (TypedEvent).
  //       We'll log the error if the event type is greater than 2 bytes.
  //       Event types are generated sequentially, so 2^16 is enough.
  MetadataRecord TypedEvent;
  TypedEvent.Type = uint8_t(RecordType::Metadata);
  TypedEvent.RecordKind =
      uint8_t(MetadataRecord::RecordKinds::TypedEventMarker);
  constexpr auto TSCSize = sizeof(TC.TSC);
  internal_memcpy(&TypedEvent.Data, &ReducedEventSize, sizeof(int32_t));
  internal_memcpy(&TypedEvent.Data[sizeof(int32_t)], &TSC, TSCSize);
  internal_memcpy(&TypedEvent.Data[sizeof(int32_t) + TSCSize], &EventType,
                  sizeof(EventType));
  internal_memcpy(TLD.RecordPtr, &TypedEvent, sizeof(TypedEvent));

  TLD.RecordPtr += sizeof(TypedEvent);
  internal_memcpy(TLD.RecordPtr, Event, ReducedEventSize);
  TLD.RecordPtr += ReducedEventSize;
  incrementExtents(MetadataRecSize + EventSize);
  endBufferIfFull();
}

XRayLogInitStatus fdrLoggingInit(UNUSED size_t BufferSize,
                                 UNUSED size_t BufferMax, void *Options,
                                 size_t OptionsSize) XRAY_NEVER_INSTRUMENT {
  if (Options == nullptr)
    return XRayLogInitStatus::XRAY_LOG_UNINITIALIZED;

  s32 CurrentStatus = XRayLogInitStatus::XRAY_LOG_UNINITIALIZED;
  if (!atomic_compare_exchange_strong(&LoggingStatus, &CurrentStatus,
                                      XRayLogInitStatus::XRAY_LOG_INITIALIZING,
                                      memory_order_release)) {
    if (Verbosity())
      Report("Cannot initialize already initialized implementation.\n");
    return static_cast<XRayLogInitStatus>(CurrentStatus);
  }

  if (Verbosity())
    Report("Initializing FDR mode with options: %s\n",
           static_cast<const char *>(Options));

  // TODO: Factor out the flags specific to the FDR mode implementation. For
  // now, use the global/single definition of the flags, since the FDR mode
  // flags are already defined there.
  FlagParser FDRParser;
  FDRFlags FDRFlags;
  registerXRayFDRFlags(&FDRParser, &FDRFlags);
  FDRFlags.setDefaults();

  // Override first from the general XRAY_DEFAULT_OPTIONS compiler-provided
  // options until we migrate everyone to use the XRAY_FDR_OPTIONS
  // compiler-provided options.
  FDRParser.ParseString(useCompilerDefinedFlags());
  FDRParser.ParseString(useCompilerDefinedFDRFlags());
  auto *EnvOpts = GetEnv("XRAY_FDR_OPTIONS");
  if (EnvOpts == nullptr)
    EnvOpts = "";
  FDRParser.ParseString(EnvOpts);

  // FIXME: Remove this when we fully remove the deprecated flags.
  if (internal_strlen(EnvOpts) == 0) {
    FDRFlags.func_duration_threshold_us =
        flags()->xray_fdr_log_func_duration_threshold_us;
    FDRFlags.grace_period_ms = flags()->xray_fdr_log_grace_period_ms;
  }

  // The provided options should always override the compiler-provided and
  // environment-variable defined options.
  FDRParser.ParseString(static_cast<const char *>(Options));
  *fdrFlags() = FDRFlags;
  BufferSize = FDRFlags.buffer_size;
  BufferMax = FDRFlags.buffer_max;

  bool Success = false;

  if (BQ != nullptr) {
    BQ->~BufferQueue();
    BQ = nullptr;
  }

  if (BQ == nullptr) {
    BQ = reinterpret_cast<BufferQueue *>(&BufferQueueStorage);
    new (BQ) BufferQueue(BufferSize, BufferMax, Success);
  }

  if (!Success) {
    Report("BufferQueue init failed.\n");
    if (BQ != nullptr) {
      BQ->~BufferQueue();
      BQ = nullptr;
    }
    return XRayLogInitStatus::XRAY_LOG_UNINITIALIZED;
  }

  static pthread_once_t OnceInit = PTHREAD_ONCE_INIT;
  pthread_once(&OnceInit, +[] {
    atomic_store(&TicksPerSec,
                 probeRequiredCPUFeatures() ? getTSCFrequency()
                                            : __xray::NanosecondsPerSecond,
                 memory_order_release);
    pthread_key_create(&Key, +[](void *TLDPtr) {
      if (TLDPtr == nullptr)
        return;
      auto &TLD = *reinterpret_cast<ThreadLocalData *>(TLDPtr);
      if (TLD.BQ == nullptr)
        return;
      auto EC = TLD.BQ->releaseBuffer(TLD.Buffer);
      if (EC != BufferQueue::ErrorCode::Ok)
        Report("At thread exit, failed to release buffer at %p; error=%s\n",
               TLD.Buffer.Data, BufferQueue::getErrorString(EC));
    });
  });

  atomic_store(&ThresholdTicks,
               atomic_load_relaxed(&TicksPerSec) *
                   fdrFlags()->func_duration_threshold_us / 1000000,
               memory_order_release);
  // Arg1 handler should go in first to avoid concurrent code accidentally
  // falling back to arg0 when it should have ran arg1.
  __xray_set_handler_arg1(fdrLoggingHandleArg1);
  // Install the actual handleArg0 handler after initialising the buffers.
  __xray_set_handler(fdrLoggingHandleArg0);
  __xray_set_customevent_handler(fdrLoggingHandleCustomEvent);
  __xray_set_typedevent_handler(fdrLoggingHandleTypedEvent);

  // Install the buffer iterator implementation.
  __xray_log_set_buffer_iterator(fdrIterator);

  atomic_store(&LoggingStatus, XRayLogInitStatus::XRAY_LOG_INITIALIZED,
               memory_order_release);

  if (Verbosity())
    Report("XRay FDR init successful.\n");
  return XRayLogInitStatus::XRAY_LOG_INITIALIZED;
}

bool fdrLogDynamicInitializer() XRAY_NEVER_INSTRUMENT {
  XRayLogImpl Impl{
      fdrLoggingInit,
      fdrLoggingFinalize,
      fdrLoggingHandleArg0,
      fdrLoggingFlush,
  };
  auto RegistrationResult = __xray_log_register_mode("xray-fdr", Impl);
  if (RegistrationResult != XRayLogRegisterStatus::XRAY_REGISTRATION_OK &&
      Verbosity()) {
    Report("Cannot register XRay FDR mode to 'xray-fdr'; error = %d\n",
           RegistrationResult);
    return false;
  }

  if (flags()->xray_fdr_log ||
      !internal_strcmp(flags()->xray_mode, "xray-fdr")) {
    auto SelectResult = __xray_log_select_mode("xray-fdr");
    if (SelectResult != XRayLogRegisterStatus::XRAY_REGISTRATION_OK &&
        Verbosity()) {
      Report("Cannot select XRay FDR mode as 'xray-fdr'; error = %d\n",
             SelectResult);
      return false;
    }
  }
  return true;
}

} // namespace __xray

static auto UNUSED Unused = __xray::fdrLogDynamicInitializer();
