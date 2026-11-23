//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "exhaustive_test.h"
#include "test/UnitTest/Test.h"

namespace {

std::atomic<uint64_t> seen[256];
uint64_t scalar_calls;

float identity(float x) {
  ++scalar_calls;
  return x;
}

LIBC_VECTOR_PCS LIBC_NAMESPACE::cpp::simd<float>
vector_identity(LIBC_NAMESPACE::cpp::simd<float> x) {
  return x;
}

using LlvmLibcMathvecMaxInputTest =
    UnaryOpChecker<float, float, identity, vector_identity>;

struct CountingChecker : public virtual LIBC_NAMESPACE::testing::Test {
  using FloatType = float;
  // Keep the full storage domain small enough for a unit test.
  using StorageType = uint8_t;
  struct FPBits {
    explicit FPBits(StorageType bits) : bits(bits) {}
    FloatType get_val() const {
      return LIBC_NAMESPACE::fputil::FPBits<FloatType>(uint32_t(bits))
          .get_val();
    }
    StorageType bits;
  };

  uint64_t check(StorageType start, StorageType stop,
                 LIBC_NAMESPACE::fputil::testing::RoundingMode) {
    for (unsigned bits = start; bits <= stop; ++bits)
      seen[bits].fetch_add(1, std::memory_order_relaxed);
    return 0;
  }
};

struct LlvmLibcMathvecRangeTest
    : public LlvmLibcExhaustiveMathvecTest<CountingChecker, 3> {
  void SetUp() override {
    for (auto &count : seen)
      count.store(0, std::memory_order_relaxed);
  }

  void expect_counts(unsigned start, unsigned stop, uint64_t count = 1) {
    for (unsigned bits = 0; bits < 256; ++bits)
      EXPECT_EQ(seen[bits].load(std::memory_order_relaxed),
                bits >= start && bits < stop ? count : uint64_t(0));
  }
};

} // namespace

TEST_F(LlvmLibcMathvecMaxInputTest, MaximumBitPatternOnce) {
  scalar_calls = 0;
  EXPECT_EQ(check(UINT32_MAX, UINT32_MAX,
                  LIBC_NAMESPACE::fputil::testing::RoundingMode::Nearest),
            uint64_t(0));
  EXPECT_EQ(scalar_calls, uint64_t(1));
}

TEST_F(LlvmLibcMathvecRangeTest, EmptyRange) {
  test_full_range_RN(7, 7);
  expect_counts(0, 0);
}

TEST_F(LlvmLibcMathvecRangeTest, SingleInput) {
  test_full_range_RN(7, 8);
  expect_counts(7, 8);
}

TEST_F(LlvmLibcMathvecRangeTest, AdjacentShards) {
  test_full_range_RN(5, 15);
  expect_counts(5, 15);
}

TEST_F(LlvmLibcMathvecRangeTest, FinalShard) {
  test_full_range_RN(250, 255);
  expect_counts(250, 255);
}

TEST_F(LlvmLibcMathvecRangeTest, FullStorageRange) {
  test_full_range_RN();
  expect_counts(0, 256);
}

TEST_F(LlvmLibcMathvecRangeTest, FullStorageRangeAllRoundings) {
  test_full_range_all_roundings();
  expect_counts(0, 256, 4);
}
