//===-- SIMDMatchers.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TEST_UNITTEST_SIMDMATCHER_H
#define LLVM_LIBC_TEST_UNITTEST_SIMDMATCHER_H

#include "hdr/stdint_proxy.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/properties/architectures.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

#include "hdr/math_macros.h"

namespace LIBC_NAMESPACE_DECL {
namespace testing {

template <typename T>
inline bool within_ulp_tolerance(T expected, T actual, uint64_t tolerance) {
  fputil::FPBits<T> expected_bits(expected), actual_bits(actual);

  if (expected_bits.is_nan() || actual_bits.is_nan())
    return expected_bits.is_nan() && actual_bits.is_nan();

  // Find the absolute difference of the input bits
  auto expected_uint = expected_bits.uintval();
  auto actual_uint = actual_bits.uintval();
  auto difference = expected_uint > actual_uint ? expected_uint - actual_uint
                                                : actual_uint - expected_uint;

  // Allow results that are within tolerance.
  return difference <= tolerance;
}

} // namespace testing
} // namespace LIBC_NAMESPACE_DECL

#define EXPECT_SIMD_EQ_EXACT(REF, RES)                                         \
  for (size_t i = 0;                                                           \
       i < LIBC_NAMESPACE::cpp::internal::native_vector_size<float>; i++) {    \
    EXPECT_FP_EQ(REF[i], RES[i]);                                              \
  }

#define EXPECT_SIMD_EQ_TOL(REF, RES, TOL)                                      \
  do {                                                                         \
    auto simd_ref = (REF);                                                     \
    auto simd_res = (RES);                                                     \
    for (size_t i = 0;                                                         \
         i < LIBC_NAMESPACE::cpp::internal::native_vector_size<float>; i++) {  \
      EXPECT_TRUE(LIBC_NAMESPACE::testing::within_ulp_tolerance(               \
          simd_ref[i], simd_res[i], (TOL)));                                   \
    }                                                                          \
  } while (0)

#define EXPECT_SIMD_EQ_SELECT(_1, _2, _3, NAME, ...) NAME
#define EXPECT_SIMD_EQ(...)                                                    \
  EXPECT_SIMD_EQ_SELECT(__VA_ARGS__, EXPECT_SIMD_EQ_TOL, EXPECT_SIMD_EQ_EXACT, \
                        unused)                                                \
  (__VA_ARGS__)

#define EXPECT_SIMD_EQ_WITH_EXCEPTION(REF, RES, EXCEPTION)                     \
  for (size_t i = 0;                                                           \
       i < LIBC_NAMESPACE::cpp::internal::native_vector_size<float>; i++) {    \
    EXPECT_FP_EQ_WITH_EXCEPTION(REF[i], RES[i], EXCEPTION);                    \
  }

#define EXPECT_SIMD_EQ_ROUNDING_MODE(expected, actual, rounding_mode)          \
  do {                                                                         \
    using namespace LIBC_NAMESPACE::fputil::testing;                           \
    ForceRoundingMode __r((rounding_mode));                                    \
    if (__r.success) {                                                         \
      EXPECT_SIMD_EQ((expected), (actual))                                     \
    }                                                                          \
  } while (0)

#define EXPECT_SIMD_EQ_ROUNDING_NEAREST(expected, actual)                      \
  EXPECT_SIMD_EQ_ROUNDING_MODE((expected), (actual), RoundingMode::Nearest)

#define EXPECT_SIMD_EQ_ROUNDING_UPWARD(expected, actual)                       \
  EXPECT_SIMD_EQ_ROUNDING_MODE((expected), (actual), RoundingMode::Upward)

#define EXPECT_SIMD_EQ_ROUNDING_DOWNWARD(expected, actual)                     \
  EXPECT_SIMD_EQ_ROUNDING_MODE((expected), (actual), RoundingMode::Downward)

#define EXPECT_SIMD_EQ_ROUNDING_TOWARD_ZERO(expected, actual)                  \
  EXPECT_SIMD_EQ_ROUNDING_MODE((expected), (actual), RoundingMode::TowardZero)

#define EXPECT_SIMD_EQ_ALL_ROUNDING(expected, actual)                          \
  do {                                                                         \
    EXPECT_SIMD_EQ_ROUNDING_NEAREST((expected), (actual));                     \
    EXPECT_SIMD_EQ_ROUNDING_UPWARD((expected), (actual));                      \
    EXPECT_SIMD_EQ_ROUNDING_DOWNWARD((expected), (actual));                    \
    EXPECT_SIMD_EQ_ROUNDING_TOWARD_ZERO((expected), (actual));                 \
  } while (0)

#endif // LLVM_LIBC_TEST_UNITTEST_SIMDMATCHER_H
