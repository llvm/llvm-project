//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Smoke tests for hypot.
///
//===----------------------------------------------------------------------===//

#include "HypotTest.h"

#include "hdr/errno_macros.h"
#include "hdr/fenv_macros.h"
#include "src/__support/macros/properties/cpu_features.h"
#include "src/math/hypot.h"

using LlvmLibcHypotTest = HypotTestTemplate<double>;

TEST_F(LlvmLibcHypotTest, SpecialNumbers) {
  test_special_numbers(&LIBC_NAMESPACE::hypot);

  // Large finite inputs (>= 2^812) whose hypot does not overflow must not raise
  // spurious FE_OVERFLOW.
  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_NEAREST(
      0x1.0000000000002p+1008,
      LIBC_NAMESPACE::hypot(0x1.78854804d5a08p+1006, 0x1.dc21e4607bbfdp+1007),
      FE_INEXACT);

  // Genuine overflow must raise FE_OVERFLOW | FE_INEXACT and set ERANGE.
  libc_errno = 0;
  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_NEAREST(
      inf, LIBC_NAMESPACE::hypot(max_normal, max_normal),
      FE_OVERFLOW | FE_INEXACT);
  EXPECT_MATH_ERRNO(ERANGE);
  libc_errno = 0;
  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_TOWARD_ZERO(
      max_normal, LIBC_NAMESPACE::hypot(max_normal, max_normal),
      FE_OVERFLOW | FE_INEXACT);
  EXPECT_MATH_ERRNO(ERANGE);

#ifndef LIBC_MATH_HAS_SKIP_ACCURATE_PASS
  // Test denormal inputs.
  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_NEAREST(
      0x0.c0bf7399534e3p-1022,
      LIBC_NAMESPACE::hypot(0x0.2c2671b3c16b3p-1022, 0x0.bb9f8fecba9adp-1022),
      FE_UNDERFLOW | FE_INEXACT);
  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_NEAREST(
      0x1.6a09e667f3bcbp-1022,
      LIBC_NAMESPACE::hypot(0x0.fffffffffffffp-1022, 0x0.fffffffffffffp-1022),
      FE_INEXACT);
  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_NEAREST(
      0x0.bffffb1b06483p-1022,
      LIBC_NAMESPACE::hypot(0x0.603e52daf0bfdp-1022, -0x0.a622d0a9a433bp-1022),
      FE_UNDERFLOW | FE_INEXACT);

  // Unrounded hypot in [2^-1022 * (1 - 2^-54), 2^-1022) rounds to min_normal in
  // 53-bit unbounded-exponent precision (not tiny after rounding, tiny before
  // rounding).
#ifdef LIBC_TARGET_CPU_HAS_TININESS_AFTER_ROUNDING
  constexpr int EXPECTED_MIN_NORMAL_EXCEPT = FE_INEXACT;
#else
  constexpr int EXPECTED_MIN_NORMAL_EXCEPT = FE_UNDERFLOW | FE_INEXACT;
#endif
  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_NEAREST(
      min_normal,
      LIBC_NAMESPACE::hypot(0x0.6ea64b75fa922p-1022, 0x0.e6da1fdfec02ep-1022),
      EXPECTED_MIN_NORMAL_EXCEPT);
  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_UPWARD(
      min_normal,
      LIBC_NAMESPACE::hypot(0x0.6ea64b75fa922p-1022, 0x0.e6da1fdfec02ep-1022),
      EXPECTED_MIN_NORMAL_EXCEPT);
  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_TOWARD_ZERO(
      max_denormal,
      LIBC_NAMESPACE::hypot(0x0.6ea64b75fa922p-1022, 0x0.e6da1fdfec02ep-1022),
      FE_UNDERFLOW | FE_INEXACT);
  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_DOWNWARD(
      max_denormal,
      LIBC_NAMESPACE::hypot(0x0.6ea64b75fa922p-1022, 0x0.e6da1fdfec02ep-1022),
      FE_UNDERFLOW | FE_INEXACT);
#endif // LIBC_MATH_HAS_SKIP_ACCURATE_PASS
}
