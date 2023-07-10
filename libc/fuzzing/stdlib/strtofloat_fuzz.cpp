//===-- strtofloat_fuzz.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Fuzzing test for llvm-libc atof implementation.
///
//===----------------------------------------------------------------------===//
#include "src/stdlib/atof.h"
#include "src/stdlib/strtod.h"
#include "src/stdlib/strtof.h"
#include "src/stdlib/strtold.h"

#include "src/__support/FPUtil/FloatProperties.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "utils/MPFRWrapper/mpfr_inc.h"

using __llvm_libc::fputil::FloatProperties;

// This function calculates the effective precision for a given float type and
// exponent. Subnormals have a lower effective precision since they don't
// necessarily use all of the bits of the mantissa.
template <typename F> inline int effective_precision(int exponent) {
  int full_precision = FloatProperties<F>::MANTISSA_PRECISION;

  // This is intended to be 0 when the exponent is the lowest normal and
  // increase as the exponent's magnitude increases.
  int bits_below_normal = (-exponent) - (FloatProperties<F>::EXPONENT_BIAS - 1);

  // This comparison is optimized out by the compiler.
  if (bits_below_normal >= 0 && bits_below_normal < full_precision - 1) {
    // The precision should be the normal, full precision, minus the bits lost
    // by this being a subnormal, minus one for the implicit leading one.
    return full_precision - bits_below_normal - 1;
  }
  return full_precision;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  uint8_t *container = new uint8_t[size + 1];
  if (!container)
    __builtin_trap();
  size_t i;

  for (i = 0; i < size; ++i) {
    // MPFR's strtofr uses "@" as a base-independent exponent symbol
    if (data[i] != '@')
      container[i] = data[i];
    else {
      container[i] = '#';
    }
  }
  container[size] = '\0'; // Add null terminator to container.

  const char *str_ptr = reinterpret_cast<const char *>(container);

  char *out_ptr = nullptr;

  size_t base = 0;

  // This is just used to determine the base and precision.
  mpfr_t result;
  mpfr_init2(result, 256);
  mpfr_t bin_result;
  mpfr_init2(bin_result, 256);
  mpfr_strtofr(result, str_ptr, &out_ptr, 0 /* base */, MPFR_RNDN);
  ptrdiff_t result_strlen = out_ptr - str_ptr;
  mpfr_strtofr(bin_result, str_ptr, &out_ptr, 2 /* base */, MPFR_RNDN);
  ptrdiff_t bin_result_strlen = out_ptr - str_ptr;

  long double bin_result_ld = mpfr_get_ld(bin_result, MPFR_RNDN);
  long double result_ld = mpfr_get_ld(result, MPFR_RNDN);

  // This detects if mpfr's strtofr selected a base of 2, which libc does not
  // support. If a base 2 decoding is detected, it is replaced by a base 10
  // decoding.
  if ((bin_result_ld != 0.0 || bin_result_strlen == result_strlen) &&
      bin_result_ld == result_ld) {
    mpfr_strtofr(result, str_ptr, &out_ptr, 10 /* base */, MPFR_RNDN);
    result_strlen = out_ptr - str_ptr;
    base = 10;
  }

  auto result_exp = mpfr_get_exp(result);

  mpfr_clear(result);
  mpfr_clear(bin_result);

  // These must be calculated with the correct precision, and not any more, to
  // prevent numbers like 66336650.00...01 (many zeroes) from causing an issue.
  // 66336650 is exactly between two float values (66336652 and 66336648) so the
  // correct float result for 66336650.00...01 is rounding up to 66336652. The
  // correct double is instead 66336650, which when converted to float is
  // rounded down to 66336648. This means we have to compare against the correct
  // precision to get the correct result.

  mpfr_t mpfr_float;
  mpfr_init2(mpfr_float, effective_precision<float>(result_exp));

  mpfr_t mpfr_double;
  mpfr_init2(mpfr_double, effective_precision<double>(result_exp));

  mpfr_t mpfr_long_double;
  mpfr_init2(mpfr_long_double, effective_precision<long double>(result_exp));

  // TODO: Add support for other rounding modes.
  mpfr_strtofr(mpfr_float, str_ptr, &out_ptr, base, MPFR_RNDN);
  mpfr_strtofr(mpfr_double, str_ptr, &out_ptr, base, MPFR_RNDN);
  mpfr_strtofr(mpfr_long_double, str_ptr, &out_ptr, base, MPFR_RNDN);

  float volatile float_result = mpfr_get_flt(mpfr_float, MPFR_RNDN);
  double volatile double_result = mpfr_get_d(mpfr_double, MPFR_RNDN);
  long double volatile long_double_result =
      mpfr_get_ld(mpfr_long_double, MPFR_RNDN);

  mpfr_clear(mpfr_float);
  mpfr_clear(mpfr_double);
  mpfr_clear(mpfr_long_double);

  auto volatile atof_output = __llvm_libc::atof(str_ptr);
  auto volatile strtof_output = __llvm_libc::strtof(str_ptr, &out_ptr);
  ptrdiff_t strtof_strlen = out_ptr - str_ptr;
  if (result_strlen != strtof_strlen)
    __builtin_trap();
  auto volatile strtod_output = __llvm_libc::strtod(str_ptr, &out_ptr);
  ptrdiff_t strtod_strlen = out_ptr - str_ptr;
  if (result_strlen != strtod_strlen)
    __builtin_trap();
  auto volatile strtold_output = __llvm_libc::strtold(str_ptr, &out_ptr);
  ptrdiff_t strtold_strlen = out_ptr - str_ptr;
  if (result_strlen != strtold_strlen)
    __builtin_trap();

  // If any result is NaN, all of them should be NaN. We can't use the usual
  // comparisons because NaN != NaN.
  if (isnan(float_result)) {
    if (!(isnan(atof_output) && isnan(strtof_output) && isnan(strtod_output) &&
          isnan(strtold_output)))
      __builtin_trap();
  } else {
    if (double_result != atof_output)
      __builtin_trap();
    if (float_result != strtof_output)
      __builtin_trap();
    if (double_result != strtod_output)
      __builtin_trap();
    if (long_double_result != strtold_output)
      __builtin_trap();
  }

  delete[] container;
  return 0;
}
