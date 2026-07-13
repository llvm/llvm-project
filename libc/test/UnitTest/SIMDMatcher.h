//===-- SIMDMatchers.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TEST_UNITTEST_SIMDMATCHER_H
#define LLVM_LIBC_TEST_UNITTEST_SIMDMATCHER_H

#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/properties/architectures.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

#include "hdr/math_macros.h"

#define EXPECT_SIMD_EQ(REF, RES)                                               \
  for (size_t i = 0;                                                           \
       i < LIBC_NAMESPACE::cpp::internal::native_vector_size<float>; i++) {    \
    EXPECT_FP_EQ(REF[i], RES[i]);                                              \
  }

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
