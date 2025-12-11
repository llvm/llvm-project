//===-- Unittests for expf ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/math_macros.h"
#include "src/__support/CPP/simd.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/math/expf.h"
#include "src/mathvec/expf.h"
#include "test/UnitTest/SIMDMatcher.h"
#include "test/UnitTest/Test.h"

#include "hdr/stdint_proxy.h"

using LlvmLibcVecExpfTest = LIBC_NAMESPACE::testing::FPTest<float>;

// Wrappers

// In order to test vector we can either duplicate a scalar input
// or do something more elaborate. In any case that requires a wrapper
// since the function call is written in this file.

// Run reference on a vector with lanes duplicated from a scalar input.

// with control lane
static LIBC_NAMESPACE::cpp::simd<float> wrap_ref_vexpf(float x, float control) {
  LIBC_NAMESPACE::cpp::simd<float> v(x);
  v[0] = control;
  constexpr size_t N = LIBC_NAMESPACE::cpp::internal::native_vector_size<float>;
  for (size_t i = 0; i < N; i++) {
    v[i] = LIBC_NAMESPACE::expf(v[i]);
  }
  return v;
}

// without control lane
static LIBC_NAMESPACE::cpp::simd<float> wrap_ref_vexpf(float x) {
  return wrap_ref_vexpf(x, x);
}

// Run implementation on a vector with lanes duplicated from a scalar input.

// with control lane
static LIBC_NAMESPACE::cpp::simd<float> wrap_vexpf(float x, float control) {
  LIBC_NAMESPACE::cpp::simd<float> v(x);
  v[0] = control;
  return LIBC_NAMESPACE::expf(v);
}

// without control lane
static LIBC_NAMESPACE::cpp::simd<float> wrap_vexpf(float x) {
  return wrap_vexpf(x, x);
}

TEST_F(LlvmLibcVecExpfTest, SpecialNumbers) {
  EXPECT_SIMD_EQ(LIBC_NAMESPACE::cpp::splat(aNaN), wrap_vexpf(aNaN));

  EXPECT_SIMD_EQ(LIBC_NAMESPACE::cpp::splat(inf), wrap_vexpf(inf));

  EXPECT_SIMD_EQ(LIBC_NAMESPACE::cpp::splat(0.0f), wrap_vexpf(neg_inf));

  EXPECT_SIMD_EQ(LIBC_NAMESPACE::cpp::splat(1.0f), wrap_vexpf(0.0f));

  EXPECT_SIMD_EQ(LIBC_NAMESPACE::cpp::splat(1.0f), wrap_vexpf(-0.0f));
}

TEST_F(LlvmLibcVecExpfTest, Overflow) {
  // Fails if tested with exceptions
  EXPECT_SIMD_EQ(LIBC_NAMESPACE::cpp::splat(inf),
                 wrap_vexpf(FPBits(0x7f7fffffU).get_val()));

  EXPECT_SIMD_EQ(LIBC_NAMESPACE::cpp::splat(inf),
                 wrap_vexpf(FPBits(0x42cffff8U).get_val()));

  EXPECT_SIMD_EQ(LIBC_NAMESPACE::cpp::splat(inf),
                 wrap_vexpf(FPBits(0x42d00008U).get_val()));
}

TEST_F(LlvmLibcVecExpfTest, Underflow) {
  // Passes if tested with exceptions ?
  EXPECT_SIMD_EQ_WITH_EXCEPTION(LIBC_NAMESPACE::cpp::splat(0.0f),
                                wrap_vexpf(FPBits(0xff7fffffU).get_val()),
                                FE_UNDERFLOW);

  float x = FPBits(0xc2cffff8U).get_val();
  EXPECT_SIMD_EQ(wrap_ref_vexpf(x, 1.0), wrap_vexpf(x, 1.0));

  x = FPBits(0xc2d00008U).get_val();
  EXPECT_SIMD_EQ(wrap_ref_vexpf(x, 1.0), wrap_vexpf(x, 1.0));
}

// Test with inputs which are the borders of underflow/overflow but still
// produce valid results without setting errno.
// Is this still relevant to vector function?
TEST_F(LlvmLibcVecExpfTest, Borderline) {
  float x;

  x = FPBits(0x42affff8U).get_val();
  // Do we need ASSERT? If so it needs a version for all rounding modes
  EXPECT_SIMD_EQ(wrap_ref_vexpf(x, 1.0), wrap_vexpf(x, 1.0));

  x = FPBits(0x42b00008U).get_val();
  EXPECT_SIMD_EQ(wrap_ref_vexpf(x, 1.0), wrap_vexpf(x, 1.0));

  x = FPBits(0xc2affff8U).get_val();
  EXPECT_SIMD_EQ(wrap_ref_vexpf(x, 1.0), wrap_vexpf(x, 1.0));

  x = FPBits(0xc2b00008U).get_val();
  EXPECT_SIMD_EQ(wrap_ref_vexpf(x, 1.0), wrap_vexpf(x, 1.0));

  x = FPBits(0xc236bd8cU).get_val();
  EXPECT_SIMD_EQ(wrap_ref_vexpf(x, 1.0), wrap_vexpf(x, 1.0));
}

TEST_F(LlvmLibcVecExpfTest, InFloatRange) {
  constexpr uint32_t COUNT = 100'000;
  constexpr uint32_t STEP = UINT32_MAX / COUNT;
  for (uint32_t i = 0, v = 0; i <= COUNT; ++i, v += STEP) {
    float x = FPBits(v).get_val();
    if (FPBits(v).is_nan() || FPBits(v).is_inf())
      continue;
    EXPECT_SIMD_EQ(wrap_ref_vexpf(x), wrap_vexpf(x));
    EXPECT_SIMD_EQ(wrap_ref_vexpf(x, aNaN), wrap_vexpf(x, aNaN));
    EXPECT_SIMD_EQ(wrap_ref_vexpf(x, inf), wrap_vexpf(x, inf));
    EXPECT_SIMD_EQ(wrap_ref_vexpf(x, -inf), wrap_vexpf(x, neg_inf));
    EXPECT_SIMD_EQ(wrap_ref_vexpf(x, 0.0), wrap_vexpf(x, 0.0));
    EXPECT_SIMD_EQ(wrap_ref_vexpf(x, -0.0), wrap_vexpf(x, -0.0));
  }
}
