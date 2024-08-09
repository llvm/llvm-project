//===-- Unittests for memory_utils ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/array.h"
#include "src/__support/macros/config.h"
#include "src/string/memory_utils/utils.h"
#include "test/UnitTest/Test.h"

namespace LIBC_NAMESPACE_DECL {

using UINT = uintptr_t;

// Converts an offset into a pointer.
const void *forge(size_t offset) {
  return reinterpret_cast<const void *>(offset);
}

TEST(LlvmLibcUtilsTest, DistanceToNextAligned) {
  EXPECT_EQ(distance_to_next_aligned<16>(forge(0)), UINT(16));
  EXPECT_EQ(distance_to_next_aligned<16>(forge(1)), UINT(15));
  EXPECT_EQ(distance_to_next_aligned<16>(forge(16)), UINT(16));
  EXPECT_EQ(distance_to_next_aligned<16>(forge(15)), UINT(1));
  EXPECT_EQ(distance_to_next_aligned<32>(forge(16)), UINT(16));
}

TEST(LlvmLibcUtilsTest, DistanceToAlignUp) {
  EXPECT_EQ(distance_to_align_up<16>(forge(0)), UINT(0));
  EXPECT_EQ(distance_to_align_up<16>(forge(1)), UINT(15));
  EXPECT_EQ(distance_to_align_up<16>(forge(16)), UINT(0));
  EXPECT_EQ(distance_to_align_up<16>(forge(15)), UINT(1));
  EXPECT_EQ(distance_to_align_up<32>(forge(16)), UINT(16));
}

TEST(LlvmLibcUtilsTest, DistanceToAlignDown) {
  EXPECT_EQ(distance_to_align_down<16>(forge(0)), UINT(0));
  EXPECT_EQ(distance_to_align_down<16>(forge(1)), UINT(1));
  EXPECT_EQ(distance_to_align_down<16>(forge(16)), UINT(0));
  EXPECT_EQ(distance_to_align_down<16>(forge(15)), UINT(15));
  EXPECT_EQ(distance_to_align_down<32>(forge(16)), UINT(16));
}

TEST(LlvmLibcUtilsTest, Adjust2) {
  char a, b;
  const size_t base_size = 10;
  for (ptrdiff_t I = -2; I < 2; ++I) {
    auto *p1 = &a;
    auto *p2 = &b;
    size_t size = base_size;
    adjust(I, p1, p2, size);
    EXPECT_EQ(intptr_t(p1), intptr_t(&a + I));
    EXPECT_EQ(intptr_t(p2), intptr_t(&b + I));
    EXPECT_EQ(size, base_size - I);
  }
}

TEST(LlvmLibcUtilsTest, Align2) {
  char a, b;
  const size_t base_size = 10;
  {
    auto *p1 = &a;
    auto *p2 = &b;
    size_t size = base_size;
    align_to_next_boundary<128, Arg::P1>(p1, p2, size);
    EXPECT_TRUE(uintptr_t(p1) % 128 == 0);
    EXPECT_GT(p1, &a);
    EXPECT_GT(p2, &b);
    EXPECT_EQ(size_t(p1 - &a), base_size - size);
    EXPECT_EQ(size_t(p2 - &b), base_size - size);
  }
  {
    auto *p1 = &a;
    auto *p2 = &b;
    size_t size = base_size;
    align_to_next_boundary<128, Arg::P2>(p1, p2, size);
    EXPECT_TRUE(uintptr_t(p2) % 128 == 0);
    EXPECT_GT(p1, &a);
    EXPECT_GT(p2, &b);
    EXPECT_EQ(size_t(p1 - &a), base_size - size);
    EXPECT_EQ(size_t(p2 - &b), base_size - size);
  }
}

TEST(LlvmLibcUtilsTest, DisjointBuffers) {
  char buf[3];
  const char *const a = buf + 0;
  const char *const b = buf + 1;
  EXPECT_TRUE(is_disjoint(a, b, 0));
  EXPECT_TRUE(is_disjoint(a, b, 1));
  EXPECT_FALSE(is_disjoint(a, b, 2));

  EXPECT_TRUE(is_disjoint(b, a, 0));
  EXPECT_TRUE(is_disjoint(b, a, 1));
  EXPECT_FALSE(is_disjoint(b, a, 2));
}

TEST(LlvmLibcUtilsTest, LoadStoreAligned) {
  const uint64_t init = 0xDEAD'C0DE'BEEF'F00D;
  CPtr const src = reinterpret_cast<CPtr>(&init);
  uint64_t store;
  Ptr const dst = reinterpret_cast<Ptr>(&store);

  using LoadFun = uint64_t (*)(CPtr);
  using StoreFun = void (*)(uint64_t, Ptr);

  {
    LoadFun ld = load_aligned<uint64_t, uint64_t>;
    StoreFun st = store_aligned<uint64_t, uint64_t>;
    const uint64_t loaded = ld(src);
    EXPECT_EQ(init, loaded);
    store = 0;
    st(init, dst);
    EXPECT_EQ(init, store);
  }

  {
    LoadFun ld = load_aligned<uint64_t, uint32_t, uint32_t>;
    StoreFun st = store_aligned<uint64_t, uint32_t, uint32_t>;
    const uint64_t loaded = ld(src);
    EXPECT_EQ(init, loaded);
    store = 0;
    st(init, dst);
    EXPECT_EQ(init, store);
  }

  {
    LoadFun ld = load_aligned<uint64_t, uint32_t, uint16_t, uint8_t, uint8_t>;
    StoreFun st = store_aligned<uint64_t, uint32_t, uint16_t, uint8_t, uint8_t>;
    const uint64_t loaded = ld(src);
    EXPECT_EQ(init, loaded);
    store = 0;
    st(init, dst);
    EXPECT_EQ(init, store);
  }
}

} // namespace LIBC_NAMESPACE_DECL
