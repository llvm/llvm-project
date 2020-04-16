//===-- Utils which wrap MPFR ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MPFRUtils.h"

#include <iostream>
#include <mpfr.h>

namespace __llvm_libc {
namespace testing {
namespace mpfr {

class MPFRNumber {
  // A precision value which allows sufficiently large additional
  // precision even compared to double precision floating point values.
  static constexpr unsigned int mpfrPrecision = 96;

  mpfr_t value;

public:
  MPFRNumber() { mpfr_init2(value, mpfrPrecision); }

  explicit MPFRNumber(float x) {
    mpfr_init2(value, mpfrPrecision);
    mpfr_set_flt(value, x, MPFR_RNDN);
  }

  MPFRNumber(const MPFRNumber &other) {
    mpfr_set(value, other.value, MPFR_RNDN);
  }

  ~MPFRNumber() { mpfr_clear(value); }

  // Returns true if |other| is within the tolerance value |t| of this
  // number.
  bool isEqual(const MPFRNumber &other, const Tolerance &t) {
    MPFRNumber tolerance(0.0);
    uint32_t bitMask = 1 << (t.width - 1);
    for (int exponent = -t.basePrecision; bitMask > 0; bitMask >>= 1) {
      --exponent;
      if (t.bits & bitMask) {
        MPFRNumber delta;
        mpfr_set_ui_2exp(delta.value, 1, exponent, MPFR_RNDN);
        mpfr_add(tolerance.value, tolerance.value, delta.value, MPFR_RNDN);
      }
    }

    MPFRNumber difference;
    if (mpfr_cmp(value, other.value) >= 0)
      mpfr_sub(difference.value, value, other.value, MPFR_RNDN);
    else
      mpfr_sub(difference.value, other.value, value, MPFR_RNDN);

    return mpfr_lessequal_p(difference.value, tolerance.value);
  }

  // These functions are useful for debugging.
  float asFloat() const { return mpfr_get_flt(value, MPFR_RNDN); }
  double asDouble() const { return mpfr_get_d(value, MPFR_RNDN); }
  void dump(const char *msg) const { mpfr_printf("%s%.128Rf\n", msg, value); }

public:
  static MPFRNumber cos(float x) {
    MPFRNumber result;
    MPFRNumber mpfrX(x);
    mpfr_cos(result.value, mpfrX.value, MPFR_RNDN);
    return result;
  }

  static MPFRNumber sin(float x) {
    MPFRNumber result;
    MPFRNumber mpfrX(x);
    mpfr_sin(result.value, mpfrX.value, MPFR_RNDN);
    return result;
  }
};

bool equalsCos(float input, float libcOutput, const Tolerance &t) {
  MPFRNumber mpfrResult = MPFRNumber::cos(input);
  MPFRNumber libcResult(libcOutput);
  return mpfrResult.isEqual(libcResult, t);
}

bool equalsSin(float input, float libcOutput, const Tolerance &t) {
  MPFRNumber mpfrResult = MPFRNumber::sin(input);
  MPFRNumber libcResult(libcOutput);
  return mpfrResult.isEqual(libcResult, t);
}

} // namespace mpfr
} // namespace testing
} // namespace __llvm_libc
