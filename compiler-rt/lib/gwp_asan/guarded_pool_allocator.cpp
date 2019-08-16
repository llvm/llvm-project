//===-- guarded_pool_allocator.cpp ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gwp_asan/guarded_pool_allocator.h"

#include "gwp_asan/options.h"

// RHEL creates the PRIu64 format macro (for printing uint64_t's) only when this
// macro is defined before including <inttypes.h>.
#ifndef __STDC_FORMAT_MACROS
  #define __STDC_FORMAT_MACROS 1
#endif

#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

using AllocationMetadata = gwp_asan::GuardedPoolAllocator::AllocationMetadata;
using Error = gwp_asan::GuardedPoolAllocator::Error;

namespace gwp_asan {
namespace {
// Forward declare the pointer to the singleton version of this class.
// Instantiated during initialisation, this allows the signal handler
// to find this class in order to deduce the root cause of failures. Must not be
// referenced by users outside this translation unit, in order to avoid
// init-order-fiasco.
GuardedPoolAllocator *SingletonPtr = nullptr;

class ScopedBoolean {
public:
  ScopedBoolean(bool &B) : Bool(B) { Bool = true; }
  ~ScopedBoolean() { Bool = false; }

private:
  bool &Bool;
};

void defaultPrintStackTrace(uintptr_t *Trace, size_t TraceLength,
                            options::Printf_t Printf) {
  if (TraceLength == 0)
    Printf("  <unknown (does your allocator support backtracing?)>\n");

  for (size_t i = 0; i < TraceLength; ++i) {
    Printf("  #%zu 0x%zx in <unknown>\n", i, Trace[i]);
  }
  Printf("\n");
}
} // anonymous namespace

// Gets the singleton implementation of this class. Thread-compatible until
// init() is called, thread-safe afterwards.
GuardedPoolAllocator *getSingleton() { return SingletonPtr; }

void GuardedPoolAllocator::AllocationMetadata::RecordAllocation(
    uintptr_t AllocAddr, size_t AllocSize, options::Backtrace_t Backtrace) {
  Addr = AllocAddr;
  Size = AllocSize;
  IsDeallocated = false;

  // TODO(hctim): Ask the caller to provide the thread ID, so we don't waste
  // other thread's time getting the thread ID under lock.
  AllocationTrace.ThreadID = getThreadID();
  AllocationTrace.TraceSize = 0;
  DeallocationTrace.TraceSize = 0;
  DeallocationTrace.ThreadID = kInvalidThreadID;

  if (Backtrace) {
    uintptr_t UncompressedBuffer[kMaxTraceLengthToCollect];
    size_t BacktraceLength =
        Backtrace(UncompressedBuffer, kMaxTraceLengthToCollect);
    AllocationTrace.TraceSize = compression::pack(
        UncompressedBuffer, BacktraceLength, AllocationTrace.CompressedTrace,
        kStackFrameStorageBytes);
  }
}

void GuardedPoolAllocator::AllocationMetadata::RecordDeallocation(
    options::Backtrace_t Backtrace) {
  IsDeallocated = true;
  // Ensure that the unwinder is not called if the recursive flag is set,
  // otherwise non-reentrant unwinders may deadlock.
  DeallocationTrace.TraceSize = 0;
  if (Backtrace && !ThreadLocals.RecursiveGuard) {
    ScopedBoolean B(ThreadLocals.RecursiveGuard);

    uintptr_t UncompressedBuffer[kMaxTraceLengthToCollect];
    size_t BacktraceLength =
        Backtrace(UncompressedBuffer, kMaxTraceLengthToCollect);
    DeallocationTrace.TraceSize = compression::pack(
        UncompressedBuffer, BacktraceLength, DeallocationTrace.CompressedTrace,
        kStackFrameStorageBytes);
  }
  DeallocationTrace.ThreadID = getThreadID();
}

void GuardedPoolAllocator::init(const options::Options &Opts) {
  // Note: We return from the constructor here if GWP-ASan is not available.
  // This will stop heap-allocation of class members, as well as mmap() of the
  // guarded slots.
  if (!Opts.Enabled || Opts.SampleRate == 0 ||
      Opts.MaxSimultaneousAllocations == 0)
    return;

  // TODO(hctim): Add a death unit test for this.
  if (SingletonPtr) {
    (*SingletonPtr->Printf)(
        "GWP-ASan Error: init() has already been called.\n");
    exit(EXIT_FAILURE);
  }

  if (Opts.SampleRate < 0) {
    Opts.Printf("GWP-ASan Error: SampleRate is < 0.\n");
    exit(EXIT_FAILURE);
  }

  if (Opts.SampleRate > INT32_MAX) {
    Opts.Printf("GWP-ASan Error: SampleRate is > 2^31.\n");
    exit(EXIT_FAILURE);
  }

  if (Opts.MaxSimultaneousAllocations < 0) {
    Opts.Printf("GWP-ASan Error: MaxSimultaneousAllocations is < 0.\n");
    exit(EXIT_FAILURE);
  }

  SingletonPtr = this;

  MaxSimultaneousAllocations = Opts.MaxSimultaneousAllocations;

  PageSize = getPlatformPageSize();

  PerfectlyRightAlign = Opts.PerfectlyRightAlign;
  Printf = Opts.Printf;
  Backtrace = Opts.Backtrace;
  if (Opts.PrintBacktrace)
    PrintBacktrace = Opts.PrintBacktrace;
  else
    PrintBacktrace = defaultPrintStackTrace;

  size_t PoolBytesRequired =
      PageSize * (1 + MaxSimultaneousAllocations) +
      MaxSimultaneousAllocations * maximumAllocationSize();
  void *GuardedPoolMemory = mapMemory(PoolBytesRequired);

  size_t BytesRequired = MaxSimultaneousAllocations * sizeof(*Metadata);
  Metadata = reinterpret_cast<AllocationMetadata *>(mapMemory(BytesRequired));
  markReadWrite(Metadata, BytesRequired);

  // Allocate memory and set up the free pages queue.
  BytesRequired = MaxSimultaneousAllocations * sizeof(*FreeSlots);
  FreeSlots = reinterpret_cast<size_t *>(mapMemory(BytesRequired));
  markReadWrite(FreeSlots, BytesRequired);

  // Multiply the sample rate by 2 to give a good, fast approximation for (1 /
  // SampleRate) chance of sampling.
  if (Opts.SampleRate != 1)
    AdjustedSampleRate = static_cast<uint32_t>(Opts.SampleRate) * 2;
  else
    AdjustedSampleRate = 1;

  GuardedPagePool = reinterpret_cast<uintptr_t>(GuardedPoolMemory);
  GuardedPagePoolEnd =
      reinterpret_cast<uintptr_t>(GuardedPoolMemory) + PoolBytesRequired;

  // Ensure that signal handlers are installed as late as possible, as the class
  // is not thread-safe until init() is finished, and thus a SIGSEGV may cause a
  // race to members if recieved during init().
  if (Opts.InstallSignalHandlers)
    installSignalHandlers();
}

void *GuardedPoolAllocator::allocate(size_t Size) {
  // GuardedPagePoolEnd == 0 when GWP-ASan is disabled. If we are disabled, fall
  // back to the supporting allocator.
  if (GuardedPagePoolEnd == 0)
    return nullptr;

  // Protect against recursivity.
  if (ThreadLocals.RecursiveGuard)
    return nullptr;
  ScopedBoolean SB(ThreadLocals.RecursiveGuard);

  if (Size == 0 || Size > maximumAllocationSize())
    return nullptr;

  size_t Index;
  {
    ScopedLock L(PoolMutex);
    Index = reserveSlot();
  }

  if (Index == kInvalidSlotID)
    return nullptr;

  uintptr_t Ptr = slotToAddr(Index);
  Ptr += allocationSlotOffset(Size);
  AllocationMetadata *Meta = addrToMetadata(Ptr);

  // If a slot is multiple pages in size, and the allocation takes up a single
  // page, we can improve overflow detection by leaving the unused pages as
  // unmapped.
  markReadWrite(reinterpret_cast<void *>(getPageAddr(Ptr)), Size);

  Meta->RecordAllocation(Ptr, Size, Backtrace);

  return reinterpret_cast<void *>(Ptr);
}

void GuardedPoolAllocator::deallocate(void *Ptr) {
  assert(pointerIsMine(Ptr) && "Pointer is not mine!");
  uintptr_t UPtr = reinterpret_cast<uintptr_t>(Ptr);
  uintptr_t SlotStart = slotToAddr(addrToSlot(UPtr));
  AllocationMetadata *Meta = addrToMetadata(UPtr);
  if (Meta->Addr != UPtr) {
    reportError(UPtr, Error::INVALID_FREE);
    exit(EXIT_FAILURE);
  }

  // Intentionally scope the mutex here, so that other threads can access the
  // pool during the expensive markInaccessible() call.
  {
    ScopedLock L(PoolMutex);
    if (Meta->IsDeallocated) {
      reportError(UPtr, Error::DOUBLE_FREE);
      exit(EXIT_FAILURE);
    }

    // Ensure that the deallocation is recorded before marking the page as
    // inaccessible. Otherwise, a racy use-after-free will have inconsistent
    // metadata.
    Meta->RecordDeallocation(Backtrace);
  }

  markInaccessible(reinterpret_cast<void *>(SlotStart),
                   maximumAllocationSize());

  // And finally, lock again to release the slot back into the pool.
  ScopedLock L(PoolMutex);
  freeSlot(addrToSlot(UPtr));
}

size_t GuardedPoolAllocator::getSize(const void *Ptr) {
  assert(pointerIsMine(Ptr));
  ScopedLock L(PoolMutex);
  AllocationMetadata *Meta = addrToMetadata(reinterpret_cast<uintptr_t>(Ptr));
  assert(Meta->Addr == reinterpret_cast<uintptr_t>(Ptr));
  return Meta->Size;
}

size_t GuardedPoolAllocator::maximumAllocationSize() const { return PageSize; }

AllocationMetadata *GuardedPoolAllocator::addrToMetadata(uintptr_t Ptr) const {
  return &Metadata[addrToSlot(Ptr)];
}

size_t GuardedPoolAllocator::addrToSlot(uintptr_t Ptr) const {
  assert(pointerIsMine(reinterpret_cast<void *>(Ptr)));
  size_t ByteOffsetFromPoolStart = Ptr - GuardedPagePool;
  return ByteOffsetFromPoolStart / (maximumAllocationSize() + PageSize);
}

uintptr_t GuardedPoolAllocator::slotToAddr(size_t N) const {
  return GuardedPagePool + (PageSize * (1 + N)) + (maximumAllocationSize() * N);
}

uintptr_t GuardedPoolAllocator::getPageAddr(uintptr_t Ptr) const {
  assert(pointerIsMine(reinterpret_cast<void *>(Ptr)));
  return Ptr & ~(static_cast<uintptr_t>(PageSize) - 1);
}

bool GuardedPoolAllocator::isGuardPage(uintptr_t Ptr) const {
  assert(pointerIsMine(reinterpret_cast<void *>(Ptr)));
  size_t PageOffsetFromPoolStart = (Ptr - GuardedPagePool) / PageSize;
  size_t PagesPerSlot = maximumAllocationSize() / PageSize;
  return (PageOffsetFromPoolStart % (PagesPerSlot + 1)) == 0;
}

size_t GuardedPoolAllocator::reserveSlot() {
  // Avoid potential reuse of a slot before we have made at least a single
  // allocation in each slot. Helps with our use-after-free detection.
  if (NumSampledAllocations < MaxSimultaneousAllocations)
    return NumSampledAllocations++;

  if (FreeSlotsLength == 0)
    return kInvalidSlotID;

  size_t ReservedIndex = getRandomUnsigned32() % FreeSlotsLength;
  size_t SlotIndex = FreeSlots[ReservedIndex];
  FreeSlots[ReservedIndex] = FreeSlots[--FreeSlotsLength];
  return SlotIndex;
}

void GuardedPoolAllocator::freeSlot(size_t SlotIndex) {
  assert(FreeSlotsLength < MaxSimultaneousAllocations);
  FreeSlots[FreeSlotsLength++] = SlotIndex;
}

uintptr_t GuardedPoolAllocator::allocationSlotOffset(size_t Size) const {
  assert(Size > 0);

  bool ShouldRightAlign = getRandomUnsigned32() % 2 == 0;
  if (!ShouldRightAlign)
    return 0;

  uintptr_t Offset = maximumAllocationSize();
  if (!PerfectlyRightAlign) {
    if (Size == 3)
      Size = 4;
    else if (Size > 4 && Size <= 8)
      Size = 8;
    else if (Size > 8 && (Size % 16) != 0)
      Size += 16 - (Size % 16);
  }
  Offset -= Size;
  return Offset;
}

void GuardedPoolAllocator::reportError(uintptr_t AccessPtr, Error E) {
  if (SingletonPtr)
    SingletonPtr->reportErrorInternal(AccessPtr, E);
}

size_t GuardedPoolAllocator::getNearestSlot(uintptr_t Ptr) const {
  if (Ptr <= GuardedPagePool + PageSize)
    return 0;
  if (Ptr > GuardedPagePoolEnd - PageSize)
    return MaxSimultaneousAllocations - 1;

  if (!isGuardPage(Ptr))
    return addrToSlot(Ptr);

  if (Ptr % PageSize <= PageSize / 2)
    return addrToSlot(Ptr - PageSize); // Round down.
  return addrToSlot(Ptr + PageSize);   // Round up.
}

Error GuardedPoolAllocator::diagnoseUnknownError(uintptr_t AccessPtr,
                                                 AllocationMetadata **Meta) {
  // Let's try and figure out what the source of this error is.
  if (isGuardPage(AccessPtr)) {
    size_t Slot = getNearestSlot(AccessPtr);
    AllocationMetadata *SlotMeta = addrToMetadata(slotToAddr(Slot));

    // Ensure that this slot was allocated once upon a time.
    if (!SlotMeta->Addr)
      return Error::UNKNOWN;
    *Meta = SlotMeta;

    if (SlotMeta->Addr < AccessPtr)
      return Error::BUFFER_OVERFLOW;
    return Error::BUFFER_UNDERFLOW;
  }

  // Access wasn't a guard page, check for use-after-free.
  AllocationMetadata *SlotMeta = addrToMetadata(AccessPtr);
  if (SlotMeta->IsDeallocated) {
    *Meta = SlotMeta;
    return Error::USE_AFTER_FREE;
  }

  // If we have reached here, the error is still unknown. There is no metadata
  // available.
  *Meta = nullptr;
  return Error::UNKNOWN;
}

namespace {
// Prints the provided error and metadata information.
void printErrorType(Error E, uintptr_t AccessPtr, AllocationMetadata *Meta,
                    options::Printf_t Printf, uint64_t ThreadID) {
  // Print using intermediate strings. Platforms like Android don't like when
  // you print multiple times to the same line, as there may be a newline
  // appended to a log file automatically per Printf() call.
  const char *ErrorString;
  switch (E) {
  case Error::UNKNOWN:
    ErrorString = "GWP-ASan couldn't automatically determine the source of "
                  "the memory error. It was likely caused by a wild memory "
                  "access into the GWP-ASan pool. The error occured";
    break;
  case Error::USE_AFTER_FREE:
    ErrorString = "Use after free";
    break;
  case Error::DOUBLE_FREE:
    ErrorString = "Double free";
    break;
  case Error::INVALID_FREE:
    ErrorString = "Invalid (wild) free";
    break;
  case Error::BUFFER_OVERFLOW:
    ErrorString = "Buffer overflow";
    break;
  case Error::BUFFER_UNDERFLOW:
    ErrorString = "Buffer underflow";
    break;
  }

  constexpr size_t kDescriptionBufferLen = 128;
  char DescriptionBuffer[kDescriptionBufferLen];
  if (Meta) {
    if (E == Error::USE_AFTER_FREE) {
      snprintf(DescriptionBuffer, kDescriptionBufferLen,
               "(%zu byte%s into a %zu-byte allocation at 0x%zx)",
               AccessPtr - Meta->Addr, (AccessPtr - Meta->Addr == 1) ? "" : "s",
               Meta->Size, Meta->Addr);
    } else if (AccessPtr < Meta->Addr) {
      snprintf(DescriptionBuffer, kDescriptionBufferLen,
               "(%zu byte%s to the left of a %zu-byte allocation at 0x%zx)",
               Meta->Addr - AccessPtr, (Meta->Addr - AccessPtr == 1) ? "" : "s",
               Meta->Size, Meta->Addr);
    } else if (AccessPtr > Meta->Addr) {
      snprintf(DescriptionBuffer, kDescriptionBufferLen,
               "(%zu byte%s to the right of a %zu-byte allocation at 0x%zx)",
               AccessPtr - Meta->Addr, (AccessPtr - Meta->Addr == 1) ? "" : "s",
               Meta->Size, Meta->Addr);
    } else {
      snprintf(DescriptionBuffer, kDescriptionBufferLen,
               "(a %zu-byte allocation)", Meta->Size);
    }
  }

  // Possible number of digits of a 64-bit number: ceil(log10(2^64)) == 20. Add
  // a null terminator, and round to the nearest 8-byte boundary.
  constexpr size_t kThreadBufferLen = 24;
  char ThreadBuffer[kThreadBufferLen];
  if (ThreadID == GuardedPoolAllocator::kInvalidThreadID)
    snprintf(ThreadBuffer, kThreadBufferLen, "<unknown>");
  else
    snprintf(ThreadBuffer, kThreadBufferLen, "%" PRIu64, ThreadID);

  Printf("%s at 0x%zx %s by thread %s here:\n", ErrorString, AccessPtr,
         DescriptionBuffer, ThreadBuffer);
}

void printAllocDeallocTraces(uintptr_t AccessPtr, AllocationMetadata *Meta,
                             options::Printf_t Printf,
                             options::PrintBacktrace_t PrintBacktrace) {
  assert(Meta != nullptr && "Metadata is non-null for printAllocDeallocTraces");

  if (Meta->IsDeallocated) {
    if (Meta->DeallocationTrace.ThreadID ==
        GuardedPoolAllocator::kInvalidThreadID)
      Printf("0x%zx was deallocated by thread <unknown> here:\n", AccessPtr);
    else
      Printf("0x%zx was deallocated by thread %zu here:\n", AccessPtr,
             Meta->DeallocationTrace.ThreadID);

    uintptr_t UncompressedTrace[AllocationMetadata::kMaxTraceLengthToCollect];
    size_t UncompressedLength = compression::unpack(
        Meta->DeallocationTrace.CompressedTrace,
        Meta->DeallocationTrace.TraceSize, UncompressedTrace,
        AllocationMetadata::kMaxTraceLengthToCollect);

    PrintBacktrace(UncompressedTrace, UncompressedLength, Printf);
  }

  if (Meta->AllocationTrace.ThreadID == GuardedPoolAllocator::kInvalidThreadID)
    Printf("0x%zx was allocated by thread <unknown> here:\n", Meta->Addr);
  else
    Printf("0x%zx was allocated by thread %zu here:\n", Meta->Addr,
           Meta->AllocationTrace.ThreadID);

  uintptr_t UncompressedTrace[AllocationMetadata::kMaxTraceLengthToCollect];
  size_t UncompressedLength = compression::unpack(
      Meta->AllocationTrace.CompressedTrace, Meta->AllocationTrace.TraceSize,
      UncompressedTrace, AllocationMetadata::kMaxTraceLengthToCollect);

  PrintBacktrace(UncompressedTrace, UncompressedLength, Printf);
}

struct ScopedEndOfReportDecorator {
  ScopedEndOfReportDecorator(options::Printf_t Printf) : Printf(Printf) {}
  ~ScopedEndOfReportDecorator() { Printf("*** End GWP-ASan report ***\n"); }
  options::Printf_t Printf;
};
} // anonymous namespace

void GuardedPoolAllocator::reportErrorInternal(uintptr_t AccessPtr, Error E) {
  if (!pointerIsMine(reinterpret_cast<void *>(AccessPtr))) {
    return;
  }

  // Attempt to prevent races to re-use the same slot that triggered this error.
  // This does not guarantee that there are no races, because another thread can
  // take the locks during the time that the signal handler is being called.
  PoolMutex.tryLock();
  ThreadLocals.RecursiveGuard = true;

  Printf("*** GWP-ASan detected a memory error ***\n");
  ScopedEndOfReportDecorator Decorator(Printf);

  AllocationMetadata *Meta = nullptr;

  if (E == Error::UNKNOWN) {
    E = diagnoseUnknownError(AccessPtr, &Meta);
  } else {
    size_t Slot = getNearestSlot(AccessPtr);
    Meta = addrToMetadata(slotToAddr(Slot));
    // Ensure that this slot has been previously allocated.
    if (!Meta->Addr)
      Meta = nullptr;
  }

  // Print the error information.
  uint64_t ThreadID = getThreadID();
  printErrorType(E, AccessPtr, Meta, Printf, ThreadID);
  if (Backtrace) {
    static constexpr unsigned kMaximumStackFramesForCrashTrace = 512;
    uintptr_t Trace[kMaximumStackFramesForCrashTrace];
    size_t TraceLength = Backtrace(Trace, kMaximumStackFramesForCrashTrace);

    PrintBacktrace(Trace, TraceLength, Printf);
  } else {
    Printf("  <unknown (does your allocator support backtracing?)>\n\n");
  }

  if (Meta)
    printAllocDeallocTraces(AccessPtr, Meta, Printf, PrintBacktrace);
}

TLS_INITIAL_EXEC
GuardedPoolAllocator::ThreadLocalPackedVariables
    GuardedPoolAllocator::ThreadLocals;
} // namespace gwp_asan
