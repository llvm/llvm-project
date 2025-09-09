//===-- Unittests for cpp::simd -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/simd.h"
#include "src/__support/CPP/utility.h"

#include "test/UnitTest/FPMatcher.h"
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
  cpp::simd<float> vf1 = cpp::splat(1.0f);
  cpp::simd<float> vf2 = cpp::splat(-1.0f);

  cpp::simd<int> v_abs = cpp::abs(vi2);
  cpp::simd<int> v_min = cpp::min(vi1, vi2);
  cpp::simd<int> v_max = cpp::max(vi1, vi2);
  cpp::simd<float> v_fma = cpp::fma(vf1, vf2, vf1);
  cpp::simd<float> v_ceil = cpp::ceil(cpp::splat(1.2f));
  cpp::simd<float> v_floor = cpp::floor(cpp::splat(1.8f));
  cpp::simd<float> v_roundeven = cpp::roundeven(cpp::splat(2.5f));
  cpp::simd<float> v_round = cpp::round(cpp::splat(2.5f));
  cpp::simd<float> v_trunc = cpp::trunc(cpp::splat(-2.9f));
  cpp::simd<float> v_nearbyint = cpp::nearbyint(cpp::splat(3.4f));
  cpp::simd<float> v_rint = cpp::rint(cpp::splat(3.6f));
  cpp::simd<float> v_canonicalize = cpp::canonicalize(cpp::splat(1.0f));
  cpp::simd<float> v_copysign = cpp::copysign(vf1, vf2);

  EXPECT_EQ(v_abs[0], 1);
  EXPECT_EQ(v_min[0], -1);
  EXPECT_EQ(v_max[0], 1);
  EXPECT_FP_EQ(v_fma[0], 0.0f);
  EXPECT_FP_EQ(v_ceil[0], 2.0f);
  EXPECT_FP_EQ(v_floor[0], 1.0f);
  EXPECT_FP_EQ(v_roundeven[0], 2.0f);
  EXPECT_FP_EQ(v_round[0], 3.0f);
  EXPECT_FP_EQ(v_trunc[0], -2.0f);
  EXPECT_FP_EQ(v_nearbyint[0], 3.0f);
  EXPECT_FP_EQ(v_rint[0], 4.0f);
  EXPECT_FP_EQ(v_canonicalize[0], 1.0f);
  EXPECT_FP_EQ(v_copysign[0], -1.0f);
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
  EXPECT_TRUE(cpp::some_of(mask));
  EXPECT_EQ(cpp::find_first_set(mask), 0);
  EXPECT_EQ(cpp::find_last_set(mask), 2);
}
