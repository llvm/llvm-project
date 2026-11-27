//===-- Unittests for cpp::simd -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/simd.h"
#include "src/__support/CPP/utility.h"

#include "test/UnitTest/Test.h"

static_assert(LIBC_HAS_VECTOR_TYPE, "compiler needs ext_vector_type support");

using namespace LIBC_NAMESPACE;

TEST(LlvmLibcSIMDTest, Basic) {}
TEST(LlvmLibcSIMDTest, VectorCreation) {
  cpp::simd<int> v1 = cpp::splat(5);
  cpp::simd<int> v2 = cpp::iota<int>();

  EXPECT_EQ(v1[0], 5);
  EXPECT_EQ(v2[0], 0);
}

TEST(LlvmLibcSIMDTest, TypeTraits) {
  cpp::simd<int> v1 = cpp::splat(0);

  static_assert(cpp::is_simd_v<decltype(v1)>, "v1 should be a SIMD type");
  static_assert(!cpp::is_simd_v<int>, "int is not a SIMD type");
  static_assert(cpp::is_simd_mask_v<cpp::simd<bool, 4>>,
                "should be a SIMD mask");

  using Elem = cpp::simd_element_type_t<decltype(v1)>;
  static_assert(cpp::is_same_v<Elem, int>, "element type should be int");
}

TEST(LlvmLibcSIMDTest, ElementwiseOperations) {
  cpp::simd<int> vi1 = cpp::splat(1);
  cpp::simd<int> vi2 = cpp::splat(-1);

  cpp::simd<int> v_abs = cpp::abs(vi2);
  cpp::simd<int> v_min = cpp::min(vi1, vi2);
  cpp::simd<int> v_max = cpp::max(vi1, vi2);

  EXPECT_EQ(v_abs[0], 1);
  EXPECT_EQ(v_min[0], -1);
  EXPECT_EQ(v_max[0], 1);
}

TEST(LlvmLibcSIMDTest, ReductionOperations) {
  cpp::simd<int> v = cpp::splat(1);

  int sum = cpp::reduce(v);
  int prod = cpp::reduce(v, cpp::multiplies<>{});

  EXPECT_EQ(sum, static_cast<int>(cpp::simd_size_v<decltype(v)>));
  EXPECT_EQ(prod, 1);
}

TEST(LlvmLibcSIMDTest, MaskOperations) {
  cpp::simd<bool, 8> mask{true, false, true, false, false, false, false, false};

  EXPECT_TRUE(cpp::any_of(mask));
  EXPECT_FALSE(cpp::all_of(mask));
  EXPECT_FALSE(cpp::none_of(mask));
  EXPECT_TRUE(cpp::some_of(mask));
  EXPECT_EQ(cpp::find_first_set(mask), 0);
  EXPECT_EQ(cpp::find_last_set(mask), 2);
  EXPECT_EQ(cpp::popcount(mask), 2);
}

TEST(LlvmLibcSIMDTest, ToBitmask) {
  cpp::simd<bool, 1> m1_false{false};
  cpp::simd<bool, 1> m1_true{true};
  EXPECT_EQ(static_cast<uint64_t>(cpp::to_bitmask(m1_false)), uint64_t{0});
  EXPECT_EQ(static_cast<uint64_t>(cpp::to_bitmask(m1_true)), uint64_t{1});

  cpp::simd<bool, 2> m2{false, true};
  EXPECT_EQ(static_cast<uint64_t>(cpp::to_bitmask(m2)), uint64_t{0b10});

  cpp::simd<bool, 4> m4{true, false, true, true};
  EXPECT_EQ(static_cast<uint64_t>(cpp::to_bitmask(m4)), uint64_t{0b1101});

  cpp::simd<bool, 8> m8{true, false, true, false, false, false, false, true};
  EXPECT_EQ(static_cast<uint64_t>(cpp::to_bitmask(m8)), uint64_t{0b10000101});

  cpp::simd<bool, 16> m16{true,  false, false, false, false, false,
                          false, false, false, false, false, false,
                          false, false, false, true};
  EXPECT_EQ(static_cast<uint64_t>(cpp::to_bitmask(m16)), uint64_t{0x8001});

  // The 24 lanes fill three bytes, but the vector occupies four. Start with
  // all ones so clearing the lanes can leave the padding byte set. Check that
  // to_bitmask still returns zero in bits 24 through 31.
  union {
    uint32_t storage;
    cpp::simd<bool, 24> mask;
  } reused{-1U};
  reused.mask = {};
  EXPECT_EQ(static_cast<uint32_t>(cpp::to_bitmask(reused.mask)), uint32_t{0});
  reused.mask[2] = true;
  reused.mask[23] = true;
  EXPECT_EQ(static_cast<uint32_t>(cpp::to_bitmask(reused.mask)),
            uint32_t{0x800004});
}

TEST(LlvmLibcSIMDTest, FromBitmask) {
  EXPECT_FALSE(cpp::from_bitmask<1>(0)[0]);
  EXPECT_TRUE(cpp::from_bitmask<1>(1)[0]);

  cpp::simd<bool, 8> mask = cpp::from_bitmask<8>(0b10000101);
  EXPECT_TRUE(mask[0]);
  EXPECT_FALSE(mask[1]);
  EXPECT_TRUE(mask[2]);
  EXPECT_FALSE(mask[3]);
  EXPECT_FALSE(mask[4]);
  EXPECT_FALSE(mask[5]);
  EXPECT_FALSE(mask[6]);
  EXPECT_TRUE(mask[7]);

  // Check bitmask that is wider than the output bool vector. The extra
  // bits at the higher end should have no effect.
  // Bits 24 through 31 are set, but only lanes 2 and 23 should be true.
  cpp::simd<bool, 24> m24 = cpp::from_bitmask<24>(0xff800004u);
  for (size_t i = 0; i < 24; ++i)
    EXPECT_EQ(m24[i], i == 2 || i == 23);
}

TEST(LlvmLibcSIMDTest, SplitConcat) {
  cpp::simd<char, 8> v{1, 1, 2, 2, 3, 3, 4, 4};
  auto [v1, v2, v3, v4] = cpp::split<2, 2, 2, 2>(v);
  EXPECT_TRUE(cpp::all_of(v1 == 1));
  EXPECT_TRUE(cpp::all_of(v2 == 2));
  EXPECT_TRUE(cpp::all_of(v3 == 3));
  EXPECT_TRUE(cpp::all_of(v4 == 4));

  cpp::simd<char, 8> m = cpp::concat(v1, v2, v3, v4);
  EXPECT_TRUE(cpp::all_of(m == v));

  cpp::simd<char, 1> c(~0);
  cpp::simd<char, 8> n = cpp::concat(c, c, c, c, c, c, c, c);
  EXPECT_TRUE(cpp::all_of(n == ~0));
}

TEST(LlvmLibcSIMDTest, LoadStore) {
  constexpr size_t SIZE = cpp::simd_size_v<cpp::simd<int>>;
  alignas(alignof(cpp::simd<int>)) int buf[SIZE];

  cpp::simd<int> v1 = cpp::splat(1);
  cpp::store(v1, buf);
  cpp::simd<int> v2 = cpp::load<cpp::simd<int>>(buf);

  EXPECT_TRUE(cpp::all_of(v1 == 1));
  EXPECT_TRUE(cpp::all_of(v2 == 1));

  cpp::simd<int> v3 = cpp::splat(2);
  cpp::store(v3, buf, /*aligned=*/true);
  cpp::simd<int> v4 = cpp::load<cpp::simd<int>>(buf, /*aligned=*/true);

  EXPECT_TRUE(cpp::all_of(v3 == 2));
  EXPECT_TRUE(cpp::all_of(v4 == 2));
}

TEST(LlvmLibcSIMDTest, MaskedLoadStore) {
  constexpr size_t SIZE = cpp::simd_size_v<cpp::simd<int>>;
  alignas(alignof(cpp::simd<int>)) int buf[SIZE] = {0};

  cpp::simd<int> mask = cpp::iota(0) % 2 == 0;
  cpp::simd<int> v1 = cpp::splat(1);

  cpp::store_masked<cpp::simd<int>>(mask, v1, buf);
  cpp::simd<int> v2 = cpp::load_masked<cpp::simd<int>>(mask, buf);

  EXPECT_TRUE(cpp::all_of((v2 == 1) == mask));
}

TEST(LlvmLibcSIMDTest, GatherScatter) {
  constexpr int SIZE = cpp::simd_size_v<cpp::simd<int>>;
  alignas(alignof(cpp::simd<int>)) int buf[SIZE];

  cpp::simd<int> mask = cpp::iota(1);
  cpp::simd<int> idx = cpp::iota(0);
  cpp::simd<int> v1 = cpp::splat(1);

  cpp::scatter<cpp::simd<int>>(mask, idx, v1, buf);
  cpp::simd<int> v2 = cpp::gather<cpp::simd<int>>(mask, idx, buf);

  EXPECT_TRUE(cpp::all_of(v1 == 1));
  EXPECT_TRUE(cpp::all_of(v2 == 1));
}

TEST(LlvmLibcSIMDTest, MaskedCompressExpand) {
  constexpr size_t SIZE = cpp::simd_size_v<cpp::simd<int>>;
  alignas(alignof(cpp::simd<int>)) int buf[SIZE] = {0};

  cpp::simd<int> mask_expand = cpp::iota(0) % 2 == 0;
  cpp::simd<int> mask_compress = 1;

  cpp::simd<int> v1 = cpp::iota(0);

  cpp::compress<cpp::simd<int>>(mask_compress, v1, buf);
  cpp::simd<int> v2 = cpp::expand<cpp::simd<int>>(mask_expand, buf);

  EXPECT_TRUE(cpp::all_of(!mask_expand || v2 <= SIZE / 2));
}
