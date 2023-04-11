//===-- Single-precision log2(x) function ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/log2f.h"
#include "common_constants.h" // Lookup table for (1/f)
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/PolyEval.h"
#include "src/__support/FPUtil/except_value_utils.h"
#include "src/__support/FPUtil/multiply_add.h"
#include "src/__support/common.h"
#include "src/__support/macros/optimization.h" // LIBC_UNLIKELY

// This is a correctly-rounded algorithm for log2(x) in single precision with
// round-to-nearest, tie-to-even mode from the RLIBM project at:
// https://people.cs.rutgers.edu/~sn349/rlibm

// Step 1 - Range reduction:
//   For x = 2^m * 1.mant, log2(x) = m + log2(1.m)
//   If x is denormal, we normalize it by multiplying x by 2^23 and subtracting
//   m by 23.

// Step 2 - Another range reduction:
//   To compute log(1.mant), let f be the highest 8 bits including the hidden
// bit, and d be the difference (1.mant - f), i.e. the remaining 16 bits of the
// mantissa. Then we have the following approximation formula:
//   log2(1.mant) = log2(f) + log2(1.mant / f)
//                = log2(f) + log2(1 + d/f)
//                ~ log2(f) + P(d/f)
// since d/f is sufficiently small.
//   log2(f) and 1/f are then stored in two 2^7 = 128 entries look-up tables.

// Step 3 - Polynomial approximation:
//   To compute P(d/f), we use a single degree-5 polynomial in double precision
// which provides correct rounding for all but few exception values.
//   For more detail about how this polynomial is obtained, please refer to the
// papers:
//   Lim, J. and Nagarakatte, S., "One Polynomial Approximation to Produce
// Correctly Rounded Results of an Elementary Function for Multiple
// Representations and Rounding Modes", Proceedings of the 49th ACM SIGPLAN
// Symposium on Principles of Programming Languages (POPL-2022), Philadelphia,
// USA, Jan. 16-22, 2022.
// https://people.cs.rutgers.edu/~sn349/papers/rlibmall-popl-2022.pdf
//   Aanjaneya, M., Lim, J., and Nagarakatte, S., "RLibm-Prog: Progressive
// Polynomial Approximations for Fast Correctly Rounded Math Libraries",
// Dept. of Comp. Sci., Rutgets U., Technical Report DCS-TR-758, Nov. 2021.
// https://arxiv.org/pdf/2111.12852.pdf.

namespace __llvm_libc {

// Lookup table for log2(f) = log2(1 + n*2^(-7)) where n = 0..127.
static constexpr double LOG2_R[128] = {
    0x0.0000000000000p+0, 0x1.72c7ba20f7327p-7, 0x1.743ee861f3556p-6,
    0x1.184b8e4c56af8p-5, 0x1.77394c9d958d5p-5, 0x1.d6ebd1f1febfep-5,
    0x1.1bb32a600549dp-4, 0x1.4c560fe68af88p-4, 0x1.7d60496cfbb4cp-4,
    0x1.960caf9abb7cap-4, 0x1.c7b528b70f1c5p-4, 0x1.f9c95dc1d1165p-4,
    0x1.097e38ce60649p-3, 0x1.22dadc2ab3497p-3, 0x1.3c6fb650cde51p-3,
    0x1.494f863b8df35p-3, 0x1.633a8bf437ce1p-3, 0x1.7046031c79f85p-3,
    0x1.8a8980abfbd32p-3, 0x1.97c1cb13c7ec1p-3, 0x1.b2602497d5346p-3,
    0x1.bfc67a7fff4ccp-3, 0x1.dac22d3e441d3p-3, 0x1.e857d3d361368p-3,
    0x1.01d9bbcfa61d4p-2, 0x1.08bce0d95fa38p-2, 0x1.169c05363f158p-2,
    0x1.1d982c9d52708p-2, 0x1.249cd2b13cd6cp-2, 0x1.32bfee370ee68p-2,
    0x1.39de8e1559f6fp-2, 0x1.4106017c3eca3p-2, 0x1.4f6fbb2cec598p-2,
    0x1.56b22e6b578e5p-2, 0x1.5dfdcf1eeae0ep-2, 0x1.6552b49986277p-2,
    0x1.6cb0f6865c8eap-2, 0x1.7b89f02cf2aadp-2, 0x1.8304d90c11fd3p-2,
    0x1.8a8980abfbd32p-2, 0x1.921800924dd3bp-2, 0x1.99b072a96c6b2p-2,
    0x1.a8ff971810a5ep-2, 0x1.b0b67f4f4681p-2,  0x1.b877c57b1b07p-2,
    0x1.c043859e2fdb3p-2, 0x1.c819dc2d45fe4p-2, 0x1.cffae611ad12bp-2,
    0x1.d7e6c0abc3579p-2, 0x1.dfdd89d586e2bp-2, 0x1.e7df5fe538ab3p-2,
    0x1.efec61b011f85p-2, 0x1.f804ae8d0cd02p-2, 0x1.0014332be0033p-1,
    0x1.042bd4b9a7c99p-1, 0x1.08494c66b8efp-1,  0x1.0c6caaf0c5597p-1,
    0x1.1096015dee4dap-1, 0x1.14c560fe68af9p-1, 0x1.18fadb6e2d3c2p-1,
    0x1.1d368296b5255p-1, 0x1.217868b0c37e8p-1, 0x1.25c0a0463bebp-1,
    0x1.2a0f3c340705cp-1, 0x1.2e644fac04fd8p-1, 0x1.2e644fac04fd8p-1,
    0x1.32bfee370ee68p-1, 0x1.37222bb70747cp-1, 0x1.3b8b1c68fa6edp-1,
    0x1.3ffad4e74f1d6p-1, 0x1.44716a2c08262p-1, 0x1.44716a2c08262p-1,
    0x1.48eef19317991p-1, 0x1.4d7380dcc422dp-1, 0x1.51ff2e30214bcp-1,
    0x1.5692101d9b4a6p-1, 0x1.5b2c3da19723bp-1, 0x1.5b2c3da19723bp-1,
    0x1.5fcdce2727ddbp-1, 0x1.6476d98ad990ap-1, 0x1.6927781d932a8p-1,
    0x1.6927781d932a8p-1, 0x1.6ddfc2a78fc63p-1, 0x1.729fd26b707c8p-1,
    0x1.7767c12967a45p-1, 0x1.7767c12967a45p-1, 0x1.7c37a9227e7fbp-1,
    0x1.810fa51bf65fdp-1, 0x1.810fa51bf65fdp-1, 0x1.85efd062c656dp-1,
    0x1.8ad846cf369a4p-1, 0x1.8ad846cf369a4p-1, 0x1.8fc924c89ac84p-1,
    0x1.94c287492c4dbp-1, 0x1.94c287492c4dbp-1, 0x1.99c48be2063c8p-1,
    0x1.9ecf50bf43f13p-1, 0x1.9ecf50bf43f13p-1, 0x1.a3e2f4ac43f6p-1,
    0x1.a8ff971810a5ep-1, 0x1.a8ff971810a5ep-1, 0x1.ae255819f022dp-1,
    0x1.b35458761d479p-1, 0x1.b35458761d479p-1, 0x1.b88cb9a2ab521p-1,
    0x1.b88cb9a2ab521p-1, 0x1.bdce9dcc96187p-1, 0x1.c31a27dd00b4ap-1,
    0x1.c31a27dd00b4ap-1, 0x1.c86f7b7ea4a89p-1, 0x1.c86f7b7ea4a89p-1,
    0x1.cdcebd2373995p-1, 0x1.d338120a6dd9dp-1, 0x1.d338120a6dd9dp-1,
    0x1.d8aba045b01c8p-1, 0x1.d8aba045b01c8p-1, 0x1.de298ec0bac0dp-1,
    0x1.de298ec0bac0dp-1, 0x1.e3b20546f554ap-1, 0x1.e3b20546f554ap-1,
    0x1.e9452c8a71028p-1, 0x1.e9452c8a71028p-1, 0x1.eee32e2aeccbfp-1,
    0x1.eee32e2aeccbfp-1, 0x1.f48c34bd1e96fp-1, 0x1.f48c34bd1e96fp-1,
    0x1.fa406bd2443dfp-1, 0x1.0000000000000p0};

LLVM_LIBC_FUNCTION(float, log2f, (float x)) {
  using FPBits = typename fputil::FPBits<float>;
  FPBits xbits(x);
  uint32_t x_u = xbits.uintval();

  // Hard to round value(s).
  using fputil::round_result_slightly_up;

  int m = -FPBits::EXPONENT_BIAS;

  // log2(1.0f) = 0.0f.
  if (LIBC_UNLIKELY(x_u == 0x3f80'0000U))
    return 0.0f;

  // Exceptional inputs.
  if (LIBC_UNLIKELY(x_u < FPBits::MIN_NORMAL || x_u > FPBits::MAX_NORMAL)) {
    if (xbits.is_zero()) {
      fputil::set_errno_if_required(ERANGE);
      fputil::raise_except_if_required(FE_DIVBYZERO);
      return static_cast<float>(FPBits::neg_inf());
    }
    if (xbits.get_sign() && !xbits.is_nan()) {
      fputil::set_errno_if_required(EDOM);
      fputil::raise_except(FE_INVALID);
      return FPBits::build_quiet_nan(0);
    }
    if (xbits.is_inf_or_nan()) {
      return x;
    }
    // Normalize denormal inputs.
    xbits.set_val(xbits.get_val() * 0x1.0p23f);
    m -= 23;
  }

  m += xbits.get_unbiased_exponent();
  int index = xbits.get_mantissa() >> 16;
  // Set bits to 1.m
  xbits.set_unbiased_exponent(0x7F);

  float u = static_cast<float>(xbits);
  double v;
#ifdef LIBC_TARGET_CPU_HAS_FMA
  v = static_cast<double>(fputil::multiply_add(u, R[index], -1.0f)); // Exact.
#else
  v = fputil::multiply_add(static_cast<double>(u), RD[index], -1.0); // Exact
#endif // LIBC_TARGET_CPU_HAS_FMA

  double extra_factor = static_cast<double>(m) + LOG2_R[index];

  // Degree-5 polynomial approximation of log2 generated by Sollya with:
  // > P = fpminimax(log2(1 + x)/x, 4, [|1, D...|], [-2^-8, 2^-7]);
  constexpr double COEFFS[5] = {0x1.71547652b8133p0, -0x1.71547652d1e33p-1,
                                0x1.ec70a098473dep-2, -0x1.7154c5ccdf121p-2,
                                0x1.2514fd90a130ap-2};

  double vsq = v * v; // Exact
  double c0 = fputil::multiply_add(v, COEFFS[0], extra_factor);
  double c1 = fputil::multiply_add(v, COEFFS[2], COEFFS[1]);
  double c2 = fputil::multiply_add(v, COEFFS[4], COEFFS[3]);

  double r = fputil::polyeval(vsq, c0, c1, c2);

  return static_cast<float>(r);
}

} // namespace __llvm_libc
