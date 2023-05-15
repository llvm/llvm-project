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

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "utils/MPFRWrapper/mpfr_inc.h"

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
  }

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
  if (isnan(result_ld)) {
    if (!(isnan(atof_output) && isnan(strtof_output) && isnan(strtod_output) &&
          isnan(strtold_output)))
      __builtin_trap();
  } else {
    if (mpfr_get_d(result, MPFR_RNDN) != atof_output)
      __builtin_trap();
    if (mpfr_get_flt(result, MPFR_RNDN) != strtof_output)
      __builtin_trap();
    if (mpfr_get_d(result, MPFR_RNDN) != strtod_output)
      __builtin_trap();
    if (mpfr_get_ld(result, MPFR_RNDN) != strtold_output)
      __builtin_trap();
  }

  mpfr_clear(result);
  mpfr_clear(bin_result);
  delete[] container;
  return 0;
}
