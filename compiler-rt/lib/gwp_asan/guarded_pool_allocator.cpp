//===-- guarded_pool_allocator.cpp ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gwp_asan/guarded_pool_allocator.h"

#include "gwp_asan/options.h"

#include <assert.h>
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
} // anonymous namespace

// Gets the singleton implementation of this class. Thread-compatible until
// init() is called, thread-safe afterwards.
GuardedPoolAllocator *getSingleton() { return SingletonPtr; }

void GuardedPoolAllocator::AllocationMetadata::RecordAllocation(
    uintptr_t AllocAddr, size_t AllocSize) {
  Addr = AllocAddr;
  Size = AllocSize;
  IsDeallocated = false;

  // TODO(hctim): Implement stack trace collection.
  // TODO(hctim): Ask the caller to provide the thread ID, so we don't waste
  // other thread's time getting the thread ID under lock.
  AllocationTrace.ThreadID = getThreadID();
  DeallocationTrace.ThreadID = kInvalidThreadID;
  AllocationTrace.Trace[0] = 0;
  DeallocationTrace.Trace[0] = 0;
}

void GuardedPoolAllocator::AllocationMetadata::RecordDeallocation() {
  IsDeallocated = true;
  // TODO(hctim): Implement stack trace collection.
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

  Meta->RecordAllocation(Ptr, Size);

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
    Meta->RecordDeallocation();
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

void GuardedPoolAllocator::reportError(uintptr_t AccessPtr, Error Error) {
  if (SingletonPtr)
    SingletonPtr->reportErrorInternal(AccessPtr, Error);
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
  return Error::UNKNOWN;
}

// Prints the provided error and metadata information. Returns true if there is
// additional context that can be provided, false otherwise (i.e. returns false
// if Error == {UNKNOWN, INVALID_FREE without metadata}).
bool printErrorType(Error Error, uintptr_t AccessPtr, AllocationMetadata *Meta,
                    options::Printf_t Printf) {
  switch (Error) {
  case Error::UNKNOWN:
    Printf("GWP-ASan couldn't automatically determine the source of the "
           "memory error when accessing 0x%zx. It was likely caused by a wild "
           "memory access into the GWP-ASan pool.\n",
           AccessPtr);
    return false;
  case Error::USE_AFTER_FREE:
    Printf("Use after free occurred when accessing memory at: 0x%zx\n",
           AccessPtr);
    break;
  case Error::DOUBLE_FREE:
    Printf("Double free occurred when trying to free memory at: 0x%zx\n",
           AccessPtr);
    break;
  case Error::INVALID_FREE:
    Printf(
        "Invalid (wild) free occurred when trying to free memory at: 0x%zx\n",
        AccessPtr);
    // It's possible for an invalid free to fall onto a slot that has never been
    // allocated. If this is the case, there is no valid metadata.
    if (Meta == nullptr)
      return false;
    break;
  case Error::BUFFER_OVERFLOW:
    Printf("Buffer overflow occurred when accessing memory at: 0x%zx\n",
           AccessPtr);
    break;
  case Error::BUFFER_UNDERFLOW:
    Printf("Buffer underflow occurred when accessing memory at: 0x%zx\n",
           AccessPtr);
    break;
  }

  Printf("0x%zx is ", AccessPtr);
  if (AccessPtr < Meta->Addr)
    Printf("located %zu bytes to the left of a %zu-byte allocation located at "
           "0x%zx\n",
           Meta->Addr - AccessPtr, Meta->Size, Meta->Addr);
  else if (AccessPtr > Meta->Addr)
    Printf("located %zu bytes to the right of a %zu-byte allocation located at "
           "0x%zx\n",
           AccessPtr - Meta->Addr, Meta->Size, Meta->Addr);
  else
    Printf("a %zu-byte allocation\n", Meta->Size);
  return true;
}

void printThreadInformation(Error Error, uintptr_t AccessPtr,
                            AllocationMetadata *Meta,
                            options::Printf_t Printf) {
  Printf("0x%zx was allocated by thread ", AccessPtr);
  if (Meta->AllocationTrace.ThreadID == UINT64_MAX)
    Printf("UNKNOWN.\n");
  else
    Printf("%zu.\n", Meta->AllocationTrace.ThreadID);

  if (Error == Error::USE_AFTER_FREE || Error == Error::DOUBLE_FREE) {
    Printf("0x%zx was freed by thread ", AccessPtr);
    if (Meta->AllocationTrace.ThreadID == UINT64_MAX)
      Printf("UNKNOWN.\n");
    else
      Printf("%zu.\n", Meta->AllocationTrace.ThreadID);
  }
}

struct ScopedEndOfReportDecorator {
  ScopedEndOfReportDecorator(options::Printf_t Printf) : Printf(Printf) {}
  ~ScopedEndOfReportDecorator() { Printf("*** End GWP-ASan report ***\n"); }
  options::Printf_t Printf;
};

void GuardedPoolAllocator::reportErrorInternal(uintptr_t AccessPtr,
                                               Error Error) {
  if (!pointerIsMine(reinterpret_cast<void *>(AccessPtr))) {
    return;
  }

  // Attempt to prevent races to re-use the same slot that triggered this error.
  // This does not guarantee that there are no races, because another thread can
  // take the locks during the time that the signal handler is being called.
  PoolMutex.tryLock();

  Printf("*** GWP-ASan detected a memory error ***\n");
  ScopedEndOfReportDecorator Decorator(Printf);

  AllocationMetadata *Meta = nullptr;

  if (Error == Error::UNKNOWN) {
    Error = diagnoseUnknownError(AccessPtr, &Meta);
  } else {
    size_t Slot = getNearestSlot(AccessPtr);
    Meta = addrToMetadata(slotToAddr(Slot));
    // Ensure that this slot has been previously allocated.
    if (!Meta->Addr)
      Meta = nullptr;
  }

  // Print the error information, and if there is no valid metadata, stop here.
  if (!printErrorType(Error, AccessPtr, Meta, Printf)) {
    return;
  }

  // Ensure that we have a valid metadata pointer from this point forward.
  if (Meta == nullptr) {
    Printf("GWP-ASan internal unreachable error. Metadata is not null.\n");
    return;
  }

  printThreadInformation(Error, AccessPtr, Meta, Printf);
  // TODO(hctim): Implement stack unwinding here. Ask the caller to provide us
  // with the base pointer, and we unwind the stack to give a stack trace for
  // the access.
  // TODO(hctim): Implement dumping here of allocation/deallocation traces.
}

TLS_INITIAL_EXEC uint64_t GuardedPoolAllocator::NextSampleCounter = 0;
} // namespace gwp_asan
