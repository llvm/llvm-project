//===-- basic.cc ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gwp_asan/tests/harness.h"

TEST_F(CustomGuardedPoolAllocator, BasicAllocation) {
  InitNumSlots(1);
  void *Ptr = GPA.allocate(1);
  EXPECT_NE(nullptr, Ptr);
  EXPECT_TRUE(GPA.pointerIsMine(Ptr));
  EXPECT_EQ(1u, GPA.getSize(Ptr));
  GPA.deallocate(Ptr);
}

TEST_F(DefaultGuardedPoolAllocator, NullptrIsNotMine) {
  EXPECT_FALSE(GPA.pointerIsMine(nullptr));
}

TEST_F(CustomGuardedPoolAllocator, SizedAllocations) {
  InitNumSlots(1);

  std::size_t MaxAllocSize = GPA.maximumAllocationSize();
  EXPECT_TRUE(MaxAllocSize > 0);

  for (unsigned AllocSize = 1; AllocSize <= MaxAllocSize; AllocSize <<= 1) {
    void *Ptr = GPA.allocate(AllocSize);
    EXPECT_NE(nullptr, Ptr);
    EXPECT_TRUE(GPA.pointerIsMine(Ptr));
    EXPECT_EQ(AllocSize, GPA.getSize(Ptr));
    GPA.deallocate(Ptr);
  }
}

TEST_F(DefaultGuardedPoolAllocator, TooLargeAllocation) {
  EXPECT_EQ(nullptr, GPA.allocate(GPA.maximumAllocationSize() + 1));
}

TEST_F(CustomGuardedPoolAllocator, AllocAllSlots) {
  constexpr unsigned kNumSlots = 128;
  InitNumSlots(kNumSlots);
  void *Ptrs[kNumSlots];
  for (unsigned i = 0; i < kNumSlots; ++i) {
    Ptrs[i] = GPA.allocate(1);
    EXPECT_NE(nullptr, Ptrs[i]);
    EXPECT_TRUE(GPA.pointerIsMine(Ptrs[i]));
  }

  // This allocation should fail as all the slots are used.
  void *Ptr = GPA.allocate(1);
  EXPECT_EQ(nullptr, Ptr);
  EXPECT_FALSE(GPA.pointerIsMine(nullptr));

  for (unsigned i = 0; i < kNumSlots; ++i)
    GPA.deallocate(Ptrs[i]);
}
