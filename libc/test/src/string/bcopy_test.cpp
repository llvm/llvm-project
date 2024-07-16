//===-- Unittests for bcopy -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/macros/config.h"
#include "src/string/bcopy.h"

#include "memory_utils/memory_check_utils.h"
#include "src/__support/CPP/span.h"
#include "test/UnitTest/MemoryMatcher.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::cpp::array;
using LIBC_NAMESPACE::cpp::span;

namespace LIBC_NAMESPACE_DECL {

TEST(LlvmLibcBcopyTest, MoveZeroByte) {
  char Buffer[] = {'a', 'b', 'y', 'z'};
  const char Expected[] = {'a', 'b', 'y', 'z'};
  void *const Dst = Buffer;
  LIBC_NAMESPACE::bcopy(Buffer + 2, Dst, 0);
  ASSERT_MEM_EQ(Buffer, testing::MemoryView(Expected));
}

TEST(LlvmLibcBcopyTest, DstAndSrcPointToSameAddress) {
  char Buffer[] = {'a', 'b'};
  const char Expected[] = {'a', 'b'};
  void *const Dst = Buffer;
  LIBC_NAMESPACE::bcopy(Buffer, Dst, 1);
  ASSERT_MEM_EQ(Buffer, testing::MemoryView(Expected));
}

TEST(LlvmLibcBcopyTest, DstStartsBeforeSrc) {
  // Set boundary at beginning and end for not overstepping when
  // copy forward or backward.
  char Buffer[] = {'z', 'a', 'b', 'c', 'z'};
  const char Expected[] = {'z', 'b', 'c', 'c', 'z'};
  void *const Dst = Buffer + 1;
  LIBC_NAMESPACE::bcopy(Buffer + 2, Dst, 2);
  ASSERT_MEM_EQ(Buffer, testing::MemoryView(Expected));
}

TEST(LlvmLibcBcopyTest, DstStartsAfterSrc) {
  char Buffer[] = {'z', 'a', 'b', 'c', 'z'};
  const char Expected[] = {'z', 'a', 'a', 'b', 'z'};
  void *const Dst = Buffer + 2;
  LIBC_NAMESPACE::bcopy(Buffer + 1, Dst, 2);
  ASSERT_MEM_EQ(Buffer, testing::MemoryView(Expected));
}

// e.g. `Dst` follow `src`.
// str: [abcdefghij]
//      [__src_____]
//      [_____Dst__]
TEST(LlvmLibcBcopyTest, SrcFollowDst) {
  char Buffer[] = {'z', 'a', 'b', 'z'};
  const char Expected[] = {'z', 'b', 'b', 'z'};
  void *const Dst = Buffer + 1;
  LIBC_NAMESPACE::bcopy(Buffer + 2, Dst, 1);
  ASSERT_MEM_EQ(Buffer, testing::MemoryView(Expected));
}

TEST(LlvmLibcBcopyTest, DstFollowSrc) {
  char Buffer[] = {'z', 'a', 'b', 'z'};
  const char Expected[] = {'z', 'a', 'a', 'z'};
  void *const Dst = Buffer + 2;
  LIBC_NAMESPACE::bcopy(Buffer + 1, Dst, 1);
  ASSERT_MEM_EQ(Buffer, testing::MemoryView(Expected));
}

// Adapt CheckMemmove signature to bcopy.
static inline void Adaptor(cpp::span<char> dst, cpp::span<char> src,
                           size_t size) {
  LIBC_NAMESPACE::bcopy(src.begin(), dst.begin(), size);
}

TEST(LlvmLibcBcopyTest, SizeSweep) {
  static constexpr int kMaxSize = 400;
  static constexpr int kDenseOverlap = 15;
  using LargeBuffer = array<char, 2 * kMaxSize + 1>;
  LargeBuffer Buffer;
  Randomize(Buffer);
  for (int Size = 0; Size < kMaxSize; ++Size)
    for (int Overlap = -1; Overlap < Size;) {
      ASSERT_TRUE(CheckMemmove<Adaptor>(Buffer, Size, Overlap));
      // Prevent quadratic behavior by skipping offset above kDenseOverlap.
      if (Overlap > kDenseOverlap)
        Overlap *= 2;
      else
        ++Overlap;
    }
}

} // namespace LIBC_NAMESPACE_DECL
