//===-- Unittests for MemorySize ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/memory_size.h"
#include "test/UnitTest/Test.h"

namespace LIBC_NAMESPACE {
namespace internal {
static inline constexpr size_t SAFE_MEM_SIZE_TEST_LIMIT =
    static_cast<size_t>(cpp::numeric_limits<cpp::make_signed_t<size_t>>::max());

TEST(LlvmLibcMemSizeTest, Constuction) {
  ASSERT_FALSE(SafeMemSize{static_cast<size_t>(-1)}.valid());
  ASSERT_FALSE(SafeMemSize{static_cast<size_t>(-2)}.valid());
  ASSERT_FALSE(SafeMemSize{static_cast<size_t>(-1024 + 33)}.valid());
  ASSERT_FALSE(SafeMemSize{static_cast<size_t>(-1024 + 66)}.valid());
  ASSERT_FALSE(SafeMemSize{SAFE_MEM_SIZE_TEST_LIMIT + 1}.valid());
  ASSERT_FALSE(SafeMemSize{SAFE_MEM_SIZE_TEST_LIMIT + 13}.valid());

  ASSERT_TRUE(SafeMemSize{static_cast<size_t>(1)}.valid());
  ASSERT_TRUE(SafeMemSize{static_cast<size_t>(1024 + 13)}.valid());
  ASSERT_TRUE(SafeMemSize{static_cast<size_t>(2048 - 13)}.valid());
  ASSERT_TRUE(SafeMemSize{static_cast<size_t>(4096 + 1)}.valid());
  ASSERT_TRUE(SafeMemSize{static_cast<size_t>(8192 - 1)}.valid());
  ASSERT_TRUE(SafeMemSize{static_cast<size_t>(16384 + 15)}.valid());
  ASSERT_TRUE(SafeMemSize{static_cast<size_t>(32768 * 3)}.valid());
  ASSERT_TRUE(SafeMemSize{static_cast<size_t>(65536 * 13)}.valid());
  ASSERT_TRUE(SafeMemSize{SAFE_MEM_SIZE_TEST_LIMIT}.valid());
  ASSERT_TRUE(SafeMemSize{SAFE_MEM_SIZE_TEST_LIMIT - 1}.valid());
  ASSERT_TRUE(SafeMemSize{SAFE_MEM_SIZE_TEST_LIMIT - 13}.valid());
}

TEST(LlvmLibcMemSizeTest, Addition) {
  auto max = SafeMemSize{SAFE_MEM_SIZE_TEST_LIMIT};
  auto half = SafeMemSize{SAFE_MEM_SIZE_TEST_LIMIT / 2};
  auto third = SafeMemSize{SAFE_MEM_SIZE_TEST_LIMIT / 3};

  ASSERT_TRUE(half.valid());
  ASSERT_TRUE(third.valid());
  ASSERT_TRUE((half + half).valid());
  ASSERT_TRUE((third + third + third).valid());
  ASSERT_TRUE((half + third).valid());

  ASSERT_FALSE((max + SafeMemSize{static_cast<size_t>(1)}).valid());
  ASSERT_FALSE((third + third + third + third).valid());
  ASSERT_FALSE((half + half + half).valid());
}

TEST(LlvmLibcMemSizeTest, Multiplication) {
  auto max = SafeMemSize{SAFE_MEM_SIZE_TEST_LIMIT};
  auto half = SafeMemSize{SAFE_MEM_SIZE_TEST_LIMIT / 2};
  auto third = SafeMemSize{SAFE_MEM_SIZE_TEST_LIMIT / 3};

  ASSERT_TRUE((max * SafeMemSize{static_cast<size_t>(1)}).valid());
  ASSERT_TRUE((max * SafeMemSize{static_cast<size_t>(0)}).valid());

  ASSERT_FALSE((max * SafeMemSize{static_cast<size_t>(2)}).valid());
  ASSERT_FALSE((half * half).valid());
  ASSERT_FALSE((half * SafeMemSize{static_cast<size_t>(3)}).valid());
  ASSERT_FALSE((third * SafeMemSize{static_cast<size_t>(4)}).valid());
}

TEST(LlvmLibcMemSizeTest, AlignUp) {
  size_t sizes[] = {
      0, 1, 8, 13, 60, 97, 128, 1024, 5124, 5120,
  };
  for (size_t i = 2; i <= 16; ++i) {
    size_t alignment = 1 << i;
    for (size_t size : sizes) {
      auto safe_size = SafeMemSize{size};
      auto safe_aligned_size = safe_size.align_up(alignment);
      ASSERT_TRUE(safe_aligned_size.valid());
      ASSERT_EQ(static_cast<size_t>(safe_aligned_size) % alignment, size_t(0));
    }
  }
  auto max = SafeMemSize{SAFE_MEM_SIZE_TEST_LIMIT};
  ASSERT_FALSE(max.align_up(8).valid());
}

TEST(LlvmLibcBlockBitTest, OffsetTo) {
  ASSERT_EQ(SafeMemSize::offset_to(0, 512), size_t(0));
  ASSERT_EQ(SafeMemSize::offset_to(1, 512), size_t(511));
  ASSERT_EQ(SafeMemSize::offset_to(2, 512), size_t(510));
  ASSERT_EQ(SafeMemSize::offset_to(13, 1), size_t(0));
  ASSERT_EQ(SafeMemSize::offset_to(13, 4), size_t(3));
  for (unsigned int i = 0; i < 31; ++i) {
    ASSERT_EQ((SafeMemSize::offset_to(i, 1u << i) + i) % (1u << i), size_t(0));
  }
}
} // namespace internal
} // namespace LIBC_NAMESPACE
