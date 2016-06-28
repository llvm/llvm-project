//===-- sanitizer_stacktrace_test.cc --------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer/AddressSanitizer runtime.
//
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_stacktrace.h"
#include "gtest/gtest.h"

namespace __sanitizer {

class FastUnwindTest : public ::testing::Test {
 protected:
  virtual void SetUp();
  virtual void TearDown();
  bool TryFastUnwind(uptr max_depth) {
    if (!StackTrace::WillUseFastUnwind(true))
      return false;
    trace.Unwind(max_depth, start_pc, (uptr)&fake_stack[0], 0, fake_top,
                 fake_bottom, true);
    return true;
  }

  void *mapping;
  uhwptr *fake_stack;
  const uptr fake_stack_size = 10;
  uhwptr start_pc;
  uhwptr fake_top;
  uhwptr fake_bottom;
  BufferedStackTrace trace;
};

static uptr PC(uptr idx) {
  return (1<<20) + idx;
}

void FastUnwindTest::SetUp() {
  size_t ps = GetPageSize();
  mapping = MmapOrDie(2 * ps, "FastUnwindTest");
  MprotectNoAccess((uptr)mapping, ps);

  // Unwinder may peek 1 word down from the starting FP.
  fake_stack = (uhwptr *)((uptr)mapping + ps + sizeof(uhwptr));

  // Fill an array of pointers with fake fp+retaddr pairs.  Frame pointers have
  // even indices.
  for (uptr i = 0; i + 1 < fake_stack_size; i += 2) {
    fake_stack[i] = (uptr)&fake_stack[i+2];  // fp
    fake_stack[i+1] = PC(i + 1); // retaddr
  }
  // Mark the last fp point back up to terminate the stack trace.
  fake_stack[RoundDownTo(fake_stack_size - 1, 2)] = (uhwptr)&fake_stack[0];

  // Top is two slots past the end because FastUnwindStack subtracts two.
  fake_top = (uhwptr)&fake_stack[fake_stack_size + 2];
  // Bottom is one slot before the start because FastUnwindStack uses >.
  fake_bottom = (uhwptr)mapping;
  start_pc = PC(0);
}

void FastUnwindTest::TearDown() {
  size_t ps = GetPageSize();
  UnmapOrDie(mapping, 2 * ps);
}

TEST_F(FastUnwindTest, Basic) {
  if (!TryFastUnwind(kStackTraceMax))
    return;
  // Should get all on-stack retaddrs and start_pc.
  EXPECT_EQ(6U, trace.size);
  EXPECT_EQ(start_pc, trace.trace[0]);
  for (uptr i = 1; i <= 5; i++) {
    EXPECT_EQ(PC(i*2 - 1), trace.trace[i]);
  }
}

// From: https://github.com/google/sanitizers/issues/162
TEST_F(FastUnwindTest, FramePointerLoop) {
  // Make one fp point to itself.
  fake_stack[4] = (uhwptr)&fake_stack[4];
  if (!TryFastUnwind(kStackTraceMax))
    return;
  // Should get all on-stack retaddrs up to the 4th slot and start_pc.
  EXPECT_EQ(4U, trace.size);
  EXPECT_EQ(start_pc, trace.trace[0]);
  for (uptr i = 1; i <= 3; i++) {
    EXPECT_EQ(PC(i*2 - 1), trace.trace[i]);
  }
}

TEST_F(FastUnwindTest, MisalignedFramePointer) {
  // Make one fp misaligned.
  fake_stack[4] += 3;
  if (!TryFastUnwind(kStackTraceMax))
    return;
  // Should get all on-stack retaddrs up to the 4th slot and start_pc.
  EXPECT_EQ(4U, trace.size);
  EXPECT_EQ(start_pc, trace.trace[0]);
  for (uptr i = 1; i < 4U; i++) {
    EXPECT_EQ(PC(i*2 - 1), trace.trace[i]);
  }
}

TEST_F(FastUnwindTest, OneFrameStackTrace) {
  if (!TryFastUnwind(1))
    return;
  EXPECT_EQ(1U, trace.size);
  EXPECT_EQ(start_pc, trace.trace[0]);
  EXPECT_EQ((uhwptr)&fake_stack[0], trace.top_frame_bp);
}

TEST_F(FastUnwindTest, ZeroFramesStackTrace) {
  if (!TryFastUnwind(0))
    return;
  EXPECT_EQ(0U, trace.size);
  EXPECT_EQ(0U, trace.top_frame_bp);
}

TEST_F(FastUnwindTest, FPBelowPrevFP) {
  // The next FP points to unreadable memory inside the stack limits, but below
  // current FP.
  fake_stack[0] = (uhwptr)&fake_stack[-50];
  fake_stack[1] = PC(1);
  if (!TryFastUnwind(3))
    return;
  EXPECT_EQ(2U, trace.size);
  EXPECT_EQ(PC(0), trace.trace[0]);
  EXPECT_EQ(PC(1), trace.trace[1]);
}

TEST_F(FastUnwindTest, CloseToZeroFrame) {
  // Make one pc a NULL pointer.
  fake_stack[5] = 0x0;
  if (!TryFastUnwind(kStackTraceMax))
    return;
  // The stack should be truncated at the NULL pointer (and not include it).
  EXPECT_EQ(3U, trace.size);
  EXPECT_EQ(start_pc, trace.trace[0]);
  for (uptr i = 1; i < 3U; i++) {
    EXPECT_EQ(PC(i*2 - 1), trace.trace[i]);
  }
}

TEST(SlowUnwindTest, ShortStackTrace) {
  if (StackTrace::WillUseFastUnwind(false))
    return;
  BufferedStackTrace stack;
  uptr pc = StackTrace::GetCurrentPc();
  uptr bp = GET_CURRENT_FRAME();
  stack.Unwind(0, pc, bp, 0, 0, 0, false);
  EXPECT_EQ(0U, stack.size);
  EXPECT_EQ(0U, stack.top_frame_bp);
  stack.Unwind(1, pc, bp, 0, 0, 0, false);
  EXPECT_EQ(1U, stack.size);
  EXPECT_EQ(pc, stack.trace[0]);
  EXPECT_EQ(bp, stack.top_frame_bp);
}

}  // namespace __sanitizer
