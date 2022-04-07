//===-- Single-precision log1p(x) function --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/log1pf.h"
#include "common_constants.h" // Lookup table for (1/f) and log(f)
#include "src/__support/FPUtil/BasicOperations.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/FMA.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/PolyEval.h"
#include "src/__support/common.h"

// This is an algorithm for log10(x) in single precision which is
// correctly rounded for all rounding modes.
// - An exhaustive test show that when x >= 2^45, log1pf(x) == logf(x)
// for all rounding modes.
// - When 2^(-8) <= |x| < 2^45, the sum (double(x) + 1.0) is exact,
// so we can adapt the correctly rounded algorithm of logf to compute
// log(double(x) + 1.0) correctly.  For more information about the logf
// algorithm, see `libc/src/math/generic/logf.cpp`.
// - When |x| < 2^(-8), we use a degree-6 polynomial in double precision
// generated with Sollya using the following command:
//   fpminimax(log(1 + x)/x, 5, [|D...|], [-2^-8; 2^-8]);

namespace __llvm_libc {

namespace internal {

// We don't need to treat denormal
INLINE_FMA static inline float log(double x) {
  constexpr double LOG_2 = 0x1.62e42fefa39efp-1;

  using FPBits = typename fputil::FPBits<double>;
  FPBits xbits(x);

  if (xbits.is_zero()) {
    return static_cast<float>(fputil::FPBits<float>::neg_inf());
  }

  if (xbits.uintval() > FPBits::MAX_NORMAL) {
    if (xbits.get_sign() && !xbits.is_nan()) {
      return fputil::FPBits<float>::build_nan(
          1 << (fputil::MantissaWidth<float>::VALUE - 1));
    }
    return static_cast<float>(x);
  }

  double m = static_cast<double>(xbits.get_exponent());

  // Set bits to 1.m
  xbits.set_unbiased_exponent(0x3FF);
  // Get the 8 highest bits, use 7 bits (excluding the implicit hidden bit) for
  // lookup tables.
  int f_index =
      xbits.get_mantissa() >> 45; // fputil::MantissaWidth<double>::VALUE - 7

  FPBits f = xbits;
  // Clear the lowest 45 bits.
  f.bits &= ~0x0000'1FFF'FFFF'FFFFULL;

  double d = static_cast<double>(xbits) - static_cast<double>(f);
  d *= ONE_OVER_F[f_index];

  double extra_factor = fputil::multiply_add(m, LOG_2, LOG_F[f_index]);

  double r = fputil::polyeval(d, extra_factor, 0x1.fffffffffffacp-1,
                              -0x1.fffffffef9cb2p-2, 0x1.5555513bc679ap-2,
                              -0x1.fff4805ea441p-3, 0x1.930180dbde91ap-3);

  return static_cast<float>(r);
}

} // namespace internal

INLINE_FMA
LLVM_LIBC_FUNCTION(float, log1pf, (float x)) {
  using FPBits = typename fputil::FPBits<float>;
  FPBits xbits(x);
  double xd = static_cast<double>(x);

  if (xbits.get_exponent() >= -8) {
    // Hard-to-round cases.
    switch (xbits.uintval()) {
    case 0x3b9315c8U: // x = 0x1.262b9p-8f
      if (fputil::get_round() != FE_UPWARD)
        return 0x1.25830cp-8f;
      break;
    case 0x3c6eb7afU: // x = 0x1.dd6f5ep-7f
      if (fputil::get_round() == FE_UPWARD)
        return 0x1.d9fd86p-7f;
      return 0x1.d9fd84p-7f;
    case 0x41078febU: // x = 0x1.0f1fd6p+3f
      if (fputil::get_round() != FE_UPWARD)
        return 0x1.1fcbcep+1f;
      break;
    case 0x5cd69e88U: // x = 0x1.ad3d1p+58f
      if (fputil::get_round() != FE_UPWARD)
        return 0x1.45c146p+5f;
      break;
    case 0x65d890d3U: // x = 0x1.b121a6p+76f
      if (fputil::get_round() == FE_TONEAREST)
        return 0x1.a9a3f2p+5f;
      break;
    case 0x6f31a8ecU: // x = 0x1.6351d8p+95f
      if (fputil::get_round() == FE_TONEAREST)
        return 0x1.08b512p+6f;
      break;
    case 0x7a17f30aU: // x = 0x1.2fe614p+117f
      if (fputil::get_round() != FE_UPWARD)
        return 0x1.451436p+6f;
      break;
    case 0xbc4d092cU: // x = -0x1.9a1258p-7f
      if (fputil::get_round() == FE_TONEAREST)
        return -0x1.9ca8bep-7f;
      break;
    case 0xbc657728U: // x = -0x1.caee5p-7f
      if (fputil::get_round() != FE_DOWNWARD)
        return -0x1.ce2cccp-7f;
      break;
    case 0xbd1d20afU: // x = -0x1.3a415ep-5f
      int round_mode = fputil::get_round();
      if (round_mode == FE_UPWARD || round_mode == FE_TOWARDZERO)
        return -0x1.40711p-5f;
      return -0x1.407112p-5f;
    }

    return internal::log(xd + 1.0);
  }

  // Hard-to round cases.
  switch (xbits.uintval()) {
  case 0x35400003U: // x = 0x1.800006p-21f
    if (fputil::get_round() == FE_TONEAREST)
      return 0x1.7ffffep-21f;
    break;
  case 0x3710001bU: // x = 0x1.200036p-17f
    if (fputil::get_round() == FE_TONEAREST)
      return 0x1.1fffe6p-17f;
    break;
  case 0xb53ffffdU: // x = -0x1.7ffffap-21f
    if (fputil::get_round() != FE_DOWNWARD)
      return -0x1.800002p-21f;
    break;
  case 0xb70fffe5U: // x = -0x1.1fffcap-17f
    if (fputil::get_round() != FE_DOWNWARD)
      return -0x1.20001ap-17f;
    break;
  case 0xbb0ec8c4U: // x = -0x1.1d9188p-9f
    if (fputil::get_round() == FE_TONEAREST)
      return -0x1.1de14ap-9f;
    break;
  }

  double r;
  // Polymial generated with Sollya:
  // > fpminimax(log(1 + x)/x, 5, [|D...|], [-2^-8; 2^-8]);
  r = fputil::polyeval(xd, -0x1p-1, 0x1.5555555515551p-2, -0x1.ffffffff82bdap-3,
                       0x1.999b33348d3aep-3, -0x1.5556cae3adcc3p-3);
  return static_cast<float>(fputil::multiply_add(r, xd * xd, xd));
}

} // namespace __llvm_libc
