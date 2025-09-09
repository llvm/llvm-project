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

using namespace LIBC_NAMESPACE::cpp;

TEST(LlvmLibcSIMDTest, Basic) {}
TEST(LlvmLibcSIMDTest, VectorCreation) {
  simd<int> v1 = splat(5);
  simd<int> v2 = iota<int>();

  EXPECT_EQ(v1[0], 5);
  EXPECT_EQ(v2[0], 0);
}

TEST(LlvmLibcSIMDTest, TypeTraits) {
  simd<int> v1 = splat(0);

  static_assert(is_simd_v<decltype(v1)>, "v1 should be a SIMD type");
  static_assert(!is_simd_v<int>, "int is not a SIMD type");
  static_assert(is_simd_mask_v<simd<bool, 4>>, "should be a SIMD mask");

  using Elem = simd_element_type_t<decltype(v1)>;
  static_assert(is_same_v<Elem, int>, "element type should be int");
}

TEST(LlvmLibcSIMDTest, ElementwiseOperations) {
  simd<int> vi1 = splat(1);
  simd<int> vi2 = splat(-1);
  simd<float> vf1 = splat(1.0f);
  simd<float> vf2 = splat(-1.0f);

  simd<int> v_abs = abs(vi2);
  simd<int> v_min = min(vi1, vi2);
  simd<int> v_max = max(vi1, vi2);
  simd<float> v_fma = fma(vf1, vf2, vf1);
  simd<float> v_ceil = ceil(splat(1.2f));
  simd<float> v_floor = floor(splat(1.8f));
  simd<float> v_roundeven = roundeven(splat(2.5f));
  simd<float> v_round = round(splat(2.5f));
  simd<float> v_trunc = trunc(splat(-2.9f));
  simd<float> v_nearbyint = nearbyint(splat(3.4f));
  simd<float> v_rint = rint(splat(3.6f));
  simd<float> v_canonicalize = canonicalize(splat(1.0f));
  simd<float> v_copysign = copysign(vf1, vf2);
  simd<float> v_fmod = fmod(splat(5.5f), splat(2.0f));

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
  EXPECT_FP_EQ(v_fmod[0], 1.5f);
}

TEST(LlvmLibcSIMDTest, ReductionOperations) {
  simd<int> v = splat(1);

  int sum = reduce(v);
  int prod = reduce(v, multiplies<>{});

  EXPECT_EQ(sum, static_cast<int>(simd_size_v<decltype(v)>));
  EXPECT_EQ(prod, 1);
}

TEST(LlvmLibcSIMDTest, MaskOperations) {
  simd<bool, 8> mask{true, false, true, false, false, false, false, false};

  EXPECT_TRUE(any_of(mask));
  EXPECT_FALSE(all_of(mask));
  EXPECT_TRUE(some_of(mask));
  EXPECT_EQ(find_first_set(mask), 0);
  EXPECT_EQ(find_last_set(mask), 2);
}
