//===-- Single-precision log1p(x) function --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/log1pf.h"
#include "common_constants.h" // Lookup table for (1/f) and log(f)
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/FMA.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/PolyEval.h"
#include "src/__support/FPUtil/except_value_utils.h"
#include "src/__support/FPUtil/multiply_add.h"
#include "src/__support/common.h"
#include "src/__support/macros/optimization.h" // LIBC_UNLIKELY
#include "src/__support/macros/properties/cpu_features.h"

// This is an algorithm for log10(x) in single precision which is
// correctly rounded for all rounding modes.
// - An exhaustive test show that when x >= 2^45, log1pf(x) == logf(x)
// for all rounding modes.
// - When 2^(-6) <= |x| < 2^45, the sum (double(x) + 1.0) is exact,
// so we can adapt the correctly rounded algorithm of logf to compute
// log(double(x) + 1.0) correctly.  For more information about the logf
// algorithm, see `libc/src/math/generic/logf.cpp`.
// - When |x| < 2^(-6), we use a degree-8 polynomial in double precision
// generated with Sollya using the following command:
//   fpminimax(log(1 + x)/x, 7, [|D...|], [-2^-6; 2^-6]);

namespace LIBC_NAMESPACE {

namespace internal {

// We don't need to treat denormal and 0
LIBC_INLINE float log(double x) {
  constexpr double LOG_2 = 0x1.62e42fefa39efp-1;

  using FPBits = typename fputil::FPBits<double>;
  FPBits xbits(x);

  uint64_t x_u = xbits.uintval();

  if (LIBC_UNLIKELY(x_u > FPBits::max_normal().uintval())) {
    if (xbits.is_neg() && !xbits.is_nan()) {
      fputil::set_errno_if_required(EDOM);
      fputil::raise_except_if_required(FE_INVALID);
      return fputil::FPBits<float>::quiet_nan().get_val();
    }
    return static_cast<float>(x);
  }

  double m = static_cast<double>(xbits.get_exponent());

  // Get the 8 highest bits, use 7 bits (excluding the implicit hidden bit) for
  // lookup tables.
  int f_index = static_cast<int>(xbits.get_mantissa() >>
                                 (fputil::FPBits<double>::FRACTION_LEN - 7));

  // Set bits to 1.m
  xbits.set_biased_exponent(0x3FF);
  FPBits f = xbits;

  // Clear the lowest 45 bits.
  f.set_uintval(f.uintval() & ~0x0000'1FFF'FFFF'FFFFULL);

  double d = xbits.get_val() - f.get_val();
  d *= ONE_OVER_F[f_index];

  double extra_factor = fputil::multiply_add(m, LOG_2, LOG_F[f_index]);

  double r = fputil::polyeval(d, extra_factor, 0x1.fffffffffffacp-1,
                              -0x1.fffffffef9cb2p-2, 0x1.5555513bc679ap-2,
                              -0x1.fff4805ea441p-3, 0x1.930180dbde91ap-3);

  return static_cast<float>(r);
}

} // namespace internal

LLVM_LIBC_FUNCTION(float, log1pf, (float x)) {
  using FPBits = typename fputil::FPBits<float>;
  FPBits xbits(x);
  uint32_t x_u = xbits.uintval();
  uint32_t x_a = x_u & 0x7fff'ffffU;
  double xd = static_cast<double>(x);

  // Use log1p(x) = log(1 + x) for |x| > 2^-6;
  if (x_a > 0x3c80'0000U) {
    // Hard-to-round cases.
    switch (x_u) {
    case 0x41078febU: // x = 0x1.0f1fd6p3
      return fputil::round_result_slightly_up(0x1.1fcbcep1f);
    case 0x5cd69e88U: // x = 0x1.ad3d1p+58f
      return fputil::round_result_slightly_up(0x1.45c146p+5f);
    case 0x65d890d3U: // x = 0x1.b121a6p+76f
      return fputil::round_result_slightly_down(0x1.a9a3f2p+5f);
    case 0x6f31a8ecU: // x = 0x1.6351d8p+95f
      return fputil::round_result_slightly_down(0x1.08b512p+6f);
    case 0x7a17f30aU: // x = 0x1.2fe614p+117f
      return fputil::round_result_slightly_up(0x1.451436p+6f);
    case 0xbd1d20afU: // x = -0x1.3a415ep-5f
      return fputil::round_result_slightly_up(-0x1.407112p-5f);
    case 0xbf800000U: // x = -1.0
      fputil::set_errno_if_required(ERANGE);
      fputil::raise_except_if_required(FE_DIVBYZERO);
      return FPBits::inf(Sign::NEG).get_val();
#ifndef LIBC_TARGET_CPU_HAS_FMA
    case 0x4cc1c80bU: // x = 0x1.839016p+26f
      return fputil::round_result_slightly_down(0x1.26fc04p+4f);
    case 0x5ee8984eU: // x = 0x1.d1309cp+62f
      return fputil::round_result_slightly_up(0x1.5c9442p+5f);
    case 0x665e7ca6U: // x = 0x1.bcf94cp+77f
      return fputil::round_result_slightly_up(0x1.af66cp+5f);
    case 0x79e7ec37U: // x = 0x1.cfd86ep+116f
      return fputil::round_result_slightly_up(0x1.43ff6ep+6f);
#endif // LIBC_TARGET_CPU_HAS_FMA
    }

    return internal::log(xd + 1.0);
  }

  // |x| <= 2^-6.
  // Hard-to round cases.
  switch (x_u) {
  case 0x35400003U: // x = 0x1.800006p-21f
    return fputil::round_result_slightly_down(0x1.7ffffep-21f);
  case 0x3710001bU: // x = 0x1.200036p-17f
    return fputil::round_result_slightly_down(0x1.1fffe6p-17f);
  case 0xb53ffffdU: // x = -0x1.7ffffap-21
    return fputil::round_result_slightly_down(-0x1.800002p-21f);
  case 0xb70fffe5U: // x = -0x1.1fffcap-17
    return fputil::round_result_slightly_down(-0x1.20001ap-17f);
  case 0xbb0ec8c4U: // x = -0x1.1d9188p-9
    return fputil::round_result_slightly_up(-0x1.1de14ap-9f);
  }

  // Polymial generated by Sollya with:
  // > fpminimax(log(1 + x)/x, 7, [|D...|], [-2^-6; 2^-6]);
  const double COEFFS[7] = {-0x1.0000000000000p-1, 0x1.5555555556aadp-2,
                            -0x1.000000000181ap-2, 0x1.999998998124ep-3,
                            -0x1.55555452e2a2bp-3, 0x1.24adb8cde4aa7p-3,
                            -0x1.0019db915ef6fp-3};

  double xsq = xd * xd;
  double c0 = fputil::multiply_add(xd, COEFFS[1], COEFFS[0]);
  double c1 = fputil::multiply_add(xd, COEFFS[3], COEFFS[2]);
  double c2 = fputil::multiply_add(xd, COEFFS[5], COEFFS[4]);
  double r = fputil::polyeval(xsq, xd, c0, c1, c2, COEFFS[6]);

  return static_cast<float>(r);
}

} // namespace LIBC_NAMESPACE
