//===-- guarded_pool_allocator.cpp ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gwp_asan/guarded_pool_allocator.h"

#include "gwp_asan/optional/segv_handler.h"
#include "gwp_asan/options.h"
#include "gwp_asan/utilities.h"

// RHEL creates the PRIu64 format macro (for printing uint64_t's) only when this
// macro is defined before including <inttypes.h>.
#ifndef __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS 1
#endif

#include <assert.h>
#include <inttypes.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

using AllocationMetadata = gwp_asan::AllocationMetadata;
using Error = gwp_asan::Error;

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
GuardedPoolAllocator *GuardedPoolAllocator::getSingleton() {
  return SingletonPtr;
}

void GuardedPoolAllocator::init(const options::Options &Opts) {
  // Note: We return from the constructor here if GWP-ASan is not available.
  // This will stop heap-allocation of class members, as well as mmap() of the
  // guarded slots.
  if (!Opts.Enabled || Opts.SampleRate == 0 ||
      Opts.MaxSimultaneousAllocations == 0)
    return;

  Check(Opts.SampleRate >= 0, "GWP-ASan Error: SampleRate is < 0.");
  Check(Opts.SampleRate < (1 << 30), "GWP-ASan Error: SampleRate is >= 2^30.");
  Check(Opts.MaxSimultaneousAllocations >= 0,
        "GWP-ASan Error: MaxSimultaneousAllocations is < 0.");

  SingletonPtr = this;
  Backtrace = Opts.Backtrace;

  State.MaxSimultaneousAllocations = Opts.MaxSimultaneousAllocations;

  State.PageSize = getPlatformPageSize();

  PerfectlyRightAlign = Opts.PerfectlyRightAlign;

  size_t PoolBytesRequired =
      State.PageSize * (1 + State.MaxSimultaneousAllocations) +
      State.MaxSimultaneousAllocations * State.maximumAllocationSize();
  void *GuardedPoolMemory = mapMemory(PoolBytesRequired, kGwpAsanGuardPageName);

  size_t BytesRequired = State.MaxSimultaneousAllocations * sizeof(*Metadata);
  Metadata = reinterpret_cast<AllocationMetadata *>(
      mapMemory(BytesRequired, kGwpAsanMetadataName));
  markReadWrite(Metadata, BytesRequired, kGwpAsanMetadataName);

  // Allocate memory and set up the free pages queue.
  BytesRequired = State.MaxSimultaneousAllocations * sizeof(*FreeSlots);
  FreeSlots = reinterpret_cast<size_t *>(
      mapMemory(BytesRequired, kGwpAsanFreeSlotsName));
  markReadWrite(FreeSlots, BytesRequired, kGwpAsanFreeSlotsName);

  // Multiply the sample rate by 2 to give a good, fast approximation for (1 /
  // SampleRate) chance of sampling.
  if (Opts.SampleRate != 1)
    AdjustedSampleRatePlusOne = static_cast<uint32_t>(Opts.SampleRate) * 2 + 1;
  else
    AdjustedSampleRatePlusOne = 2;

  initPRNG();
  ThreadLocals.NextSampleCounter =
      (getRandomUnsigned32() % (AdjustedSampleRatePlusOne - 1)) + 1;

  State.GuardedPagePool = reinterpret_cast<uintptr_t>(GuardedPoolMemory);
  State.GuardedPagePoolEnd =
      reinterpret_cast<uintptr_t>(GuardedPoolMemory) + PoolBytesRequired;

  if (Opts.InstallForkHandlers)
    installAtFork();
}

void GuardedPoolAllocator::disable() { PoolMutex.lock(); }

void GuardedPoolAllocator::enable() { PoolMutex.unlock(); }

void GuardedPoolAllocator::iterate(void *Base, size_t Size, iterate_callback Cb,
                                   void *Arg) {
  uintptr_t Start = reinterpret_cast<uintptr_t>(Base);
  for (size_t i = 0; i < State.MaxSimultaneousAllocations; ++i) {
    const AllocationMetadata &Meta = Metadata[i];
    if (Meta.Addr && !Meta.IsDeallocated && Meta.Addr >= Start &&
        Meta.Addr < Start + Size)
      Cb(Meta.Addr, Meta.Size, Arg);
  }
}

void GuardedPoolAllocator::uninitTestOnly() {
  if (State.GuardedPagePool) {
    unmapMemory(reinterpret_cast<void *>(State.GuardedPagePool),
                State.GuardedPagePoolEnd - State.GuardedPagePool,
                kGwpAsanGuardPageName);
    State.GuardedPagePool = 0;
    State.GuardedPagePoolEnd = 0;
  }
  if (Metadata) {
    unmapMemory(Metadata, State.MaxSimultaneousAllocations * sizeof(*Metadata),
                kGwpAsanMetadataName);
    Metadata = nullptr;
  }
  if (FreeSlots) {
    unmapMemory(FreeSlots,
                State.MaxSimultaneousAllocations * sizeof(*FreeSlots),
                kGwpAsanFreeSlotsName);
    FreeSlots = nullptr;
  }
}

static uintptr_t getPageAddr(uintptr_t Ptr, uintptr_t PageSize) {
  return Ptr & ~(PageSize - 1);
}

void *GuardedPoolAllocator::allocate(size_t Size) {
  // GuardedPagePoolEnd == 0 when GWP-ASan is disabled. If we are disabled, fall
  // back to the supporting allocator.
  if (State.GuardedPagePoolEnd == 0) {
    ThreadLocals.NextSampleCounter = AdjustedSampleRatePlusOne - 1;
    return nullptr;
  }

  // Protect against recursivity.
  if (ThreadLocals.RecursiveGuard)
    return nullptr;
  ScopedRecursiveGuard SRG;

  if (Size == 0 || Size > State.maximumAllocationSize())
    return nullptr;

  size_t Index;
  {
    ScopedLock L(PoolMutex);
    Index = reserveSlot();
  }

  if (Index == kInvalidSlotID)
    return nullptr;

  uintptr_t Ptr = State.slotToAddr(Index);
  // Should we right-align this allocation?
  if (getRandomUnsigned32() % 2 == 0) {
    AlignmentStrategy Align = AlignmentStrategy::DEFAULT;
    if (PerfectlyRightAlign)
      Align = AlignmentStrategy::PERFECT;
    Ptr +=
        State.maximumAllocationSize() - rightAlignedAllocationSize(Size, Align);
  }
  AllocationMetadata *Meta = addrToMetadata(Ptr);

  // If a slot is multiple pages in size, and the allocation takes up a single
  // page, we can improve overflow detection by leaving the unused pages as
  // unmapped.
  markReadWrite(reinterpret_cast<void *>(getPageAddr(Ptr, State.PageSize)),
                Size, kGwpAsanAliveSlotName);

  Meta->RecordAllocation(Ptr, Size);
  Meta->AllocationTrace.RecordBacktrace(Backtrace);

  return reinterpret_cast<void *>(Ptr);
}

void GuardedPoolAllocator::trapOnAddress(uintptr_t Address, Error E) {
  State.FailureType = E;
  State.FailureAddress = Address;

  // Raise a SEGV by touching first guard page.
  volatile char *p = reinterpret_cast<char *>(State.GuardedPagePool);
  *p = 0;
  __builtin_unreachable();
}

void GuardedPoolAllocator::stop() {
  ThreadLocals.RecursiveGuard = true;
  PoolMutex.tryLock();
}

void GuardedPoolAllocator::deallocate(void *Ptr) {
  assert(pointerIsMine(Ptr) && "Pointer is not mine!");
  uintptr_t UPtr = reinterpret_cast<uintptr_t>(Ptr);
  size_t Slot = State.getNearestSlot(UPtr);
  uintptr_t SlotStart = State.slotToAddr(Slot);
  AllocationMetadata *Meta = addrToMetadata(UPtr);
  if (Meta->Addr != UPtr) {
    // If multiple errors occur at the same time, use the first one.
    ScopedLock L(PoolMutex);
    trapOnAddress(UPtr, Error::INVALID_FREE);
  }

  // Intentionally scope the mutex here, so that other threads can access the
  // pool during the expensive markInaccessible() call.
  {
    ScopedLock L(PoolMutex);
    if (Meta->IsDeallocated) {
      trapOnAddress(UPtr, Error::DOUBLE_FREE);
    }

    // Ensure that the deallocation is recorded before marking the page as
    // inaccessible. Otherwise, a racy use-after-free will have inconsistent
    // metadata.
    Meta->RecordDeallocation();

    // Ensure that the unwinder is not called if the recursive flag is set,
    // otherwise non-reentrant unwinders may deadlock.
    if (!ThreadLocals.RecursiveGuard) {
      ScopedRecursiveGuard SRG;
      Meta->DeallocationTrace.RecordBacktrace(Backtrace);
    }
  }

  markInaccessible(reinterpret_cast<void *>(SlotStart),
                   State.maximumAllocationSize(), kGwpAsanGuardPageName);

  // And finally, lock again to release the slot back into the pool.
  ScopedLock L(PoolMutex);
  freeSlot(Slot);
}

size_t GuardedPoolAllocator::getSize(const void *Ptr) {
  assert(pointerIsMine(Ptr));
  ScopedLock L(PoolMutex);
  AllocationMetadata *Meta = addrToMetadata(reinterpret_cast<uintptr_t>(Ptr));
  assert(Meta->Addr == reinterpret_cast<uintptr_t>(Ptr));
  return Meta->Size;
}

AllocationMetadata *GuardedPoolAllocator::addrToMetadata(uintptr_t Ptr) const {
  return &Metadata[State.getNearestSlot(Ptr)];
}

size_t GuardedPoolAllocator::reserveSlot() {
  // Avoid potential reuse of a slot before we have made at least a single
  // allocation in each slot. Helps with our use-after-free detection.
  if (NumSampledAllocations < State.MaxSimultaneousAllocations)
    return NumSampledAllocations++;

  if (FreeSlotsLength == 0)
    return kInvalidSlotID;

  size_t ReservedIndex = getRandomUnsigned32() % FreeSlotsLength;
  size_t SlotIndex = FreeSlots[ReservedIndex];
  FreeSlots[ReservedIndex] = FreeSlots[--FreeSlotsLength];
  return SlotIndex;
}

void GuardedPoolAllocator::freeSlot(size_t SlotIndex) {
  assert(FreeSlotsLength < State.MaxSimultaneousAllocations);
  FreeSlots[FreeSlotsLength++] = SlotIndex;
}

uint32_t GuardedPoolAllocator::getRandomUnsigned32() {
  uint32_t RandomState = ThreadLocals.RandomState;
  RandomState ^= RandomState << 13;
  RandomState ^= RandomState >> 17;
  RandomState ^= RandomState << 5;
  ThreadLocals.RandomState = RandomState;
  return RandomState;
}

GWP_ASAN_TLS_INITIAL_EXEC
GuardedPoolAllocator::ThreadLocalPackedVariables
    GuardedPoolAllocator::ThreadLocals;
} // namespace gwp_asan
