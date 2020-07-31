//===-- backtrace.cpp -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <string>

#include "gwp_asan/common.h"
#include "gwp_asan/crash_handler.h"
#include "gwp_asan/tests/harness.h"

// Optnone to ensure that the calls to these functions are not optimized away,
// as we're looking for them in the backtraces.
__attribute((optnone)) void *
AllocateMemory(gwp_asan::GuardedPoolAllocator &GPA) {
  return GPA.allocate(1);
}
__attribute((optnone)) void
DeallocateMemory(gwp_asan::GuardedPoolAllocator &GPA, void *Ptr) {
  GPA.deallocate(Ptr);
}
__attribute((optnone)) void
DeallocateMemory2(gwp_asan::GuardedPoolAllocator &GPA, void *Ptr) {
  GPA.deallocate(Ptr);
}
__attribute__((optnone)) void TouchMemory(void *Ptr) {
  *(reinterpret_cast<volatile char *>(Ptr)) = 7;
}

TEST_F(BacktraceGuardedPoolAllocator, DoubleFree) {
  void *Ptr = AllocateMemory(GPA);
  DeallocateMemory(GPA, Ptr);

  std::string DeathRegex = "Double Free.*";
  DeathRegex.append("DeallocateMemory2.*");

  DeathRegex.append("was deallocated.*");
  DeathRegex.append("DeallocateMemory.*");

  DeathRegex.append("was allocated.*");
  DeathRegex.append("AllocateMemory.*");
  ASSERT_DEATH(DeallocateMemory2(GPA, Ptr), DeathRegex);
}

TEST_F(BacktraceGuardedPoolAllocator, UseAfterFree) {
  void *Ptr = AllocateMemory(GPA);
  DeallocateMemory(GPA, Ptr);

  std::string DeathRegex = "Use After Free.*";
  DeathRegex.append("TouchMemory.*");

  DeathRegex.append("was deallocated.*");
  DeathRegex.append("DeallocateMemory.*");

  DeathRegex.append("was allocated.*");
  DeathRegex.append("AllocateMemory.*");
  ASSERT_DEATH(TouchMemory(Ptr), DeathRegex);
}

TEST(Backtrace, Short) {
  gwp_asan::AllocationMetadata Meta;
  Meta.AllocationTrace.RecordBacktrace(
      [](uintptr_t *TraceBuffer, size_t /* Size */) -> size_t {
        TraceBuffer[0] = 123u;
        TraceBuffer[1] = 321u;
        return 2u;
      });
  uintptr_t TraceOutput[2] = {};
  EXPECT_EQ(2u, __gwp_asan_get_allocation_trace(&Meta, TraceOutput, 2));
  EXPECT_EQ(TraceOutput[0], 123u);
  EXPECT_EQ(TraceOutput[1], 321u);
}

TEST(Backtrace, ExceedsStorableLength) {
  gwp_asan::AllocationMetadata Meta;
  Meta.AllocationTrace.RecordBacktrace(
      [](uintptr_t *TraceBuffer, size_t Size) -> size_t {
        // Need to inintialise the elements that will be packed.
        memset(TraceBuffer, 0u, Size * sizeof(*TraceBuffer));

        // Indicate that there were more frames, and we just didn't have enough
        // room to store them.
        return Size * 2;
      });
  // Retrieve a frame from the collected backtrace, make sure it works E2E.
  uintptr_t TraceOutput;
  EXPECT_EQ(gwp_asan::AllocationMetadata::kMaxTraceLengthToCollect,
            __gwp_asan_get_allocation_trace(&Meta, &TraceOutput, 1));
}

TEST(Backtrace, ExceedsRetrievableAllocLength) {
  gwp_asan::AllocationMetadata Meta;
  constexpr size_t kNumFramesToStore = 3u;
  Meta.AllocationTrace.RecordBacktrace(
      [](uintptr_t *TraceBuffer, size_t /* Size */) -> size_t {
        memset(TraceBuffer, kNumFramesToStore,
               kNumFramesToStore * sizeof(*TraceBuffer));
        return kNumFramesToStore;
      });
  uintptr_t TraceOutput;
  // Ask for one element, get told that there's `kNumFramesToStore` available.
  EXPECT_EQ(kNumFramesToStore,
            __gwp_asan_get_allocation_trace(&Meta, &TraceOutput, 1));
}

TEST(Backtrace, ExceedsRetrievableDeallocLength) {
  gwp_asan::AllocationMetadata Meta;
  constexpr size_t kNumFramesToStore = 3u;
  Meta.DeallocationTrace.RecordBacktrace(
      [](uintptr_t *TraceBuffer, size_t /* Size */) -> size_t {
        memset(TraceBuffer, kNumFramesToStore,
               kNumFramesToStore * sizeof(*TraceBuffer));
        return kNumFramesToStore;
      });
  uintptr_t TraceOutput;
  // Ask for one element, get told that there's `kNumFramesToStore` available.
  EXPECT_EQ(kNumFramesToStore,
            __gwp_asan_get_deallocation_trace(&Meta, &TraceOutput, 1));
}
