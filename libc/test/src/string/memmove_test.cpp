//===-- Unittests for memmove ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/span.h"
#include "src/string/memmove.h"
#include "utils/UnitTest/MemoryMatcher.h"
#include "utils/UnitTest/Test.h"

using __llvm_libc::cpp::array;
using __llvm_libc::cpp::span;

TEST(LlvmLibcMemmoveTest, MoveZeroByte) {
  char Buffer[] = {'a', 'b', 'y', 'z'};
  const char Expected[] = {'a', 'b', 'y', 'z'};
  void *const Dst = Buffer;
  void *const Ret = __llvm_libc::memmove(Dst, Buffer + 2, 0);
  EXPECT_EQ(Ret, Dst);
  ASSERT_MEM_EQ(Buffer, Expected);
}

TEST(LlvmLibcMemmoveTest, DstAndSrcPointToSameAddress) {
  char Buffer[] = {'a', 'b'};
  const char Expected[] = {'a', 'b'};
  void *const Dst = Buffer;
  void *const Ret = __llvm_libc::memmove(Dst, Buffer, 1);
  EXPECT_EQ(Ret, Dst);
  ASSERT_MEM_EQ(Buffer, Expected);
}

TEST(LlvmLibcMemmoveTest, DstStartsBeforeSrc) {
  // Set boundary at beginning and end for not overstepping when
  // copy forward or backward.
  char Buffer[] = {'z', 'a', 'b', 'c', 'z'};
  const char Expected[] = {'z', 'b', 'c', 'c', 'z'};
  void *const Dst = Buffer + 1;
  void *const Ret = __llvm_libc::memmove(Dst, Buffer + 2, 2);
  EXPECT_EQ(Ret, Dst);
  ASSERT_MEM_EQ(Buffer, Expected);
}

TEST(LlvmLibcMemmoveTest, DstStartsAfterSrc) {
  char Buffer[] = {'z', 'a', 'b', 'c', 'z'};
  const char Expected[] = {'z', 'a', 'a', 'b', 'z'};
  void *const Dst = Buffer + 2;
  void *const Ret = __llvm_libc::memmove(Dst, Buffer + 1, 2);
  EXPECT_EQ(Ret, Dst);
  ASSERT_MEM_EQ(Buffer, Expected);
}

// e.g. `Dst` follow `src`.
// str: [abcdefghij]
//      [__src_____]
//      [_____Dst__]
TEST(LlvmLibcMemmoveTest, SrcFollowDst) {
  char Buffer[] = {'z', 'a', 'b', 'z'};
  const char Expected[] = {'z', 'b', 'b', 'z'};
  void *const Dst = Buffer + 1;
  void *const Ret = __llvm_libc::memmove(Dst, Buffer + 2, 1);
  EXPECT_EQ(Ret, Dst);
  ASSERT_MEM_EQ(Buffer, Expected);
}

TEST(LlvmLibcMemmoveTest, DstFollowSrc) {
  char Buffer[] = {'z', 'a', 'b', 'z'};
  const char Expected[] = {'z', 'a', 'a', 'z'};
  void *const Dst = Buffer + 2;
  void *const Ret = __llvm_libc::memmove(Dst, Buffer + 1, 1);
  EXPECT_EQ(Ret, Dst);
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

TEST(LlvmLibcMemmoveTest, SizeSweep) {
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
      void *const Ret =
          __llvm_libc::memmove(Dst, Buffer.data() + SrcOffset, Size);
      EXPECT_EQ(Ret, Dst);
      ASSERT_MEM_EQ(Buffer, Expected);
    }
  }
}
