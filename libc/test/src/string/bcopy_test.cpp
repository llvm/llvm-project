//===-- Unittests for bcopy -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/span.h"
#include "src/string/bcopy.h"
#include "utils/UnitTest/MemoryMatcher.h"
#include "utils/UnitTest/Test.h"

using __llvm_libc::cpp::array;
using __llvm_libc::cpp::span;

TEST(LlvmLibcBcopyTest, MoveZeroByte) {
  char Buffer[] = {'a', 'b', 'y', 'z'};
  const char Expected[] = {'a', 'b', 'y', 'z'};
  void *const Dst = Buffer;
  __llvm_libc::bcopy(Buffer + 2, Dst, 0);
  ASSERT_MEM_EQ(Buffer, Expected);
}

TEST(LlvmLibcBcopyTest, DstAndSrcPointToSameAddress) {
  char Buffer[] = {'a', 'b'};
  const char Expected[] = {'a', 'b'};
  void *const Dst = Buffer;
  __llvm_libc::bcopy(Buffer, Dst, 1);
  ASSERT_MEM_EQ(Buffer, Expected);
}

TEST(LlvmLibcBcopyTest, DstStartsBeforeSrc) {
  // Set boundary at beginning and end for not overstepping when
  // copy forward or backward.
  char Buffer[] = {'z', 'a', 'b', 'c', 'z'};
  const char Expected[] = {'z', 'b', 'c', 'c', 'z'};
  void *const Dst = Buffer + 1;
  __llvm_libc::bcopy(Buffer + 2, Dst, 2);
  ASSERT_MEM_EQ(Buffer, Expected);
}

TEST(LlvmLibcBcopyTest, DstStartsAfterSrc) {
  char Buffer[] = {'z', 'a', 'b', 'c', 'z'};
  const char Expected[] = {'z', 'a', 'a', 'b', 'z'};
  void *const Dst = Buffer + 2;
  __llvm_libc::bcopy(Buffer + 1, Dst, 2);
  ASSERT_MEM_EQ(Buffer, Expected);
}

// e.g. `Dst` follow `src`.
// str: [abcdefghij]
//      [__src_____]
//      [_____Dst__]
TEST(LlvmLibcBcopyTest, SrcFollowDst) {
  char Buffer[] = {'z', 'a', 'b', 'z'};
  const char Expected[] = {'z', 'b', 'b', 'z'};
  void *const Dst = Buffer + 1;
  __llvm_libc::bcopy(Buffer + 2, Dst, 1);
  ASSERT_MEM_EQ(Buffer, Expected);
}

TEST(LlvmLibcBcopyTest, DstFollowSrc) {
  char Buffer[] = {'z', 'a', 'b', 'z'};
  const char Expected[] = {'z', 'a', 'a', 'z'};
  void *const Dst = Buffer + 2;
  __llvm_libc::bcopy(Buffer + 1, Dst, 1);
  ASSERT_MEM_EQ(Buffer, Expected);
}

static constexpr int kMaxSize = 512;

char GetRandomChar() {
  static constexpr const uint64_t A = 1103515245;
  static constexpr const uint64_t C = 12345;
  static constexpr const uint64_t M = 1ULL << 31;
  static uint64_t Seed = 123456789;
  Seed = (A * Seed + C) % M;
  return Seed;
}

void Randomize(span<char> Buffer) {
  for (auto &current : Buffer)
    current = GetRandomChar();
}

TEST(LlvmLibcBcopyTest, SizeSweep) {
  using LargeBuffer = array<char, 3 * kMaxSize>;
  LargeBuffer GroundTruth;
  Randomize(GroundTruth);
  for (int Size = 0; Size < kMaxSize; ++Size) {
    for (int Offset = -Size; Offset < Size; ++Offset) {
      LargeBuffer Buffer = GroundTruth;
      LargeBuffer Expected = GroundTruth;
      size_t DstOffset = kMaxSize;
      size_t SrcOffset = kMaxSize + Offset;
      for (int I = 0; I < Size; ++I)
        Expected[DstOffset + I] = GroundTruth[SrcOffset + I];
      void *const Dst = Buffer.data() + DstOffset;
      __llvm_libc::bcopy(Buffer.data() + SrcOffset, Dst, Size);
      ASSERT_MEM_EQ(Buffer, Expected);
    }
  }
}
