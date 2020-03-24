//===-- backtrace.cpp -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <string>

#include "gwp_asan/crash_handler.h"
#include "gwp_asan/tests/harness.h"

TEST_F(BacktraceGuardedPoolAllocator, DoubleFree) {
  void *Ptr = GPA.allocate(1);
  GPA.deallocate(Ptr);

  std::string DeathRegex = "Double Free.*";
  DeathRegex.append("backtrace\\.cpp:26.*");

  DeathRegex.append("was deallocated.*");
  DeathRegex.append("backtrace\\.cpp:16.*");

  DeathRegex.append("was allocated.*");
  DeathRegex.append("backtrace\\.cpp:15.*");
  ASSERT_DEATH(GPA.deallocate(Ptr), DeathRegex);
}

TEST_F(BacktraceGuardedPoolAllocator, UseAfterFree) {
  char *Ptr = static_cast<char *>(GPA.allocate(1));
  GPA.deallocate(Ptr);

  std::string DeathRegex = "Use After Free.*";
  DeathRegex.append("backtrace\\.cpp:41.*");

  DeathRegex.append("was deallocated.*");
  DeathRegex.append("backtrace\\.cpp:31.*");

  DeathRegex.append("was allocated.*");
  DeathRegex.append("backtrace\\.cpp:30.*");
  ASSERT_DEATH({ *Ptr = 7; }, DeathRegex);
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
      [](uintptr_t * /* TraceBuffer */, size_t /* Size */) -> size_t {
        return SIZE_MAX; // Wow, that's big!
      });
  uintptr_t TraceOutput;
  EXPECT_EQ(1u, __gwp_asan_get_allocation_trace(&Meta, &TraceOutput, 1));
}
