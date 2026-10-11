//===-- Unittests for the alignment of memory from new --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/new.h"
#include "src/__support/alloc-checker.h"
#include "test/UnitTest/Test.h"

#include <stdint.h>

using LIBC_NAMESPACE::AllocChecker;

// The compiler may assume that memory from new is aligned for the largest
// scalar type and zero it with aligned vector stores. The allocator the
// hermetic tests link, which hands out consecutive blocks, has to honour that,
// so a block that follows a small one must not land on an 8 byte boundary.
TEST(LlvmLibcNewAlignmentTest, BlocksFollowingSmallOnesAreAligned) {
  constexpr uintptr_t ALIGNMENT = __BIGGEST_ALIGNMENT__;
  AllocChecker ac;
  for (int i = 0; i < 4; ++i) {
    uint64_t *block = new (ac) uint64_t;
    ASSERT_TRUE(static_cast<bool>(ac));
    EXPECT_EQ(reinterpret_cast<uintptr_t>(block) % ALIGNMENT, uintptr_t(0));
    delete block;
  }
}
