//===-- Benchmark Memory Test ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LibcMemoryBenchmark.h"
#include "llvm/Support/Alignment.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using testing::AllOf;
using testing::AnyOf;
using testing::ElementsAre;
using testing::Ge;
using testing::Gt;
using testing::Le;
using testing::Lt;

namespace llvm {
namespace libc_benchmarks {
namespace {

TEST(AlignedBuffer, IsAligned) {
  AlignedBuffer AB(0);
  EXPECT_TRUE(isAddrAligned(Align(AlignedBuffer::Alignment), AB.begin()));
}

TEST(AlignedBuffer, Empty) {
  AlignedBuffer AB(0);
  EXPECT_EQ(std::distance(AB.begin(), AB.end()), 0U);
}

TEST(OffsetDistribution, AlignToBegin) {
  const size_t BufferSize = 8192;
  OffsetDistribution OD(BufferSize, 1024, None);
  std::default_random_engine Gen;
  for (size_t I = 0; I <= 10; ++I)
    EXPECT_EQ(OD(Gen), 0U);
}

TEST(OffsetDistribution, NoAlignment) {
  const size_t BufferSize = 8192;
  OffsetDistribution OD(BufferSize, 1, Align(1));
  std::default_random_engine Gen;
  for (size_t I = 0; I <= 10; ++I)
    EXPECT_THAT(OD(Gen), AllOf(Ge(0U), Lt(8192U)));
}

MATCHER_P(IsDivisibleBy, n, "") {
  *result_listener << "where the remainder is " << (arg % n);
  return (arg % n) == 0;
}

TEST(OffsetDistribution, Aligned) {
  const size_t BufferSize = 8192;
  OffsetDistribution OD(BufferSize, 1, Align(16));
  std::default_random_engine Gen;
  for (size_t I = 0; I <= 10; ++I)
    EXPECT_THAT(OD(Gen), AllOf(Ge(0U), Lt(8192U), IsDivisibleBy(16U)));
}

TEST(MismatchOffsetDistribution, EqualBufferDisablesDistribution) {
  const size_t BufferSize = 8192;
  const uint32_t MismatchAt = 0; // buffer are equal.

  MismatchOffsetDistribution MOD(BufferSize, 1024, MismatchAt);
  EXPECT_FALSE(MOD);
}

TEST(MismatchOffsetDistribution, DifferentBufferDisablesDistribution) {
  const size_t BufferSize = 8192;
  const uint32_t MismatchAt = 1; // buffer are different.

  MismatchOffsetDistribution MOD(BufferSize, 1024, MismatchAt);
  EXPECT_FALSE(MOD);
}

TEST(MismatchOffsetDistribution, MismatchAt2) {
  const size_t BufferSize = 16;
  const uint32_t MismatchAt = 2; // buffer are different at position 2.
  const uint32_t MaxSize = 4;

  MismatchOffsetDistribution MOD(BufferSize, MaxSize, MismatchAt);
  EXPECT_TRUE(MOD);
  // We test equality up to MaxSize (=4) so we need spans of 4 equal bytes
  // spaced by one mismatch.
  EXPECT_THAT(MOD.getMismatchIndices(), ElementsAre(5, 9, 13));
  std::default_random_engine Gen;
  for (size_t Iterations = 0; Iterations <= 10; ++Iterations) {
    for (size_t Size = 0; Size <= MaxSize; ++Size) {
      if (Size >= MismatchAt)
        EXPECT_THAT(MOD(Gen, Size),
                    AnyOf(5 - MismatchAt, 9 - MismatchAt, 13 - MismatchAt));
      else
        EXPECT_THAT(MOD(Gen, Size),
                    AnyOf(5 - Size - 1, 9 - Size - 1, 13 - Size - 1));
    }
  }
}

} // namespace
} // namespace libc_benchmarks
} // namespace llvm
