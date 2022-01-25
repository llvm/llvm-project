//===-- Single-precision log(x) function ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/logf.h"
#include "common_constants.h" // Lookup table for (1/f)
#include "src/__support/FPUtil/BasicOperations.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/FMA.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/PolyEval.h"
#include "src/__support/common.h"

// This is an algorithm for log(x) in single precision which is correctly
// rounded for all rounding modes, based on the implementation of log(x) from
// the RLIBM project at:
// https://people.cs.rutgers.edu/~sn349/rlibm

// Step 1 - Range reduction:
//   For x = 2^m * 1.mant, log(x) = m * log(2) + log(1.m)
//   If x is denormal, we normalize it by multiplying x by 2^23 and subtracting
//   m by 23.

// Step 2 - Another range reduction:
//   To compute log(1.mant), let f be the highest 8 bits including the hidden
// bit, and d be the difference (1.mant - f), i.e. the remaining 16 bits of the
// mantissa. Then we have the following approximation formula:
//   log(1.mant) = log(f) + log(1.mant / f)
//               = log(f) + log(1 + d/f)
//               ~ log(f) + P(d/f)
// since d/f is sufficiently small.
//   log(f) and 1/f are then stored in two 2^7 = 128 entries look-up tables.

// Step 3 - Polynomial approximation:
//   To compute P(d/f), we use a single degree-5 polynomial in double precision
// which provides correct rounding for all but few exception values.
//   For more detail about how this polynomial is obtained, please refer to the
// paper:
//   Lim, J. and Nagarakatte, S., "One Polynomial Approximation to Produce
// Correctly Rounded Results of an Elementary Function for Multiple
// Representations and Rounding Modes", Proceedings of the 49th ACM SIGPLAN
// Symposium on Principles of Programming Languages (POPL-2022), Philadelphia,
// USA, January 16-22, 2022.
// https://people.cs.rutgers.edu/~sn349/papers/rlibmall-popl-2022.pdf

namespace __llvm_libc {

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc++17-extensions"

// Lookup table for log(f) = log(1 + n*2^(-7)) where n = 0..127.
static constexpr double LOG_F[128] = {
    0x0.0000000000000p+0, 0x1.fe02a6b106788p-8, 0x1.fc0a8b0fc03e3p-7,
    0x1.7b91b07d5b11ap-6, 0x1.f829b0e783300p-6, 0x1.39e87b9febd5fp-5,
    0x1.77458f632dcfcp-5, 0x1.b42dd711971bep-5, 0x1.f0a30c01162a6p-5,
    0x1.16536eea37ae0p-4, 0x1.341d7961bd1d0p-4, 0x1.51b073f06183fp-4,
    0x1.6f0d28ae56b4bp-4, 0x1.8c345d6319b20p-4, 0x1.a926d3a4ad563p-4,
    0x1.c5e548f5bc743p-4, 0x1.e27076e2af2e5p-4, 0x1.fec9131dbeabap-4,
    0x1.0d77e7cd08e59p-3, 0x1.1b72ad52f67a0p-3, 0x1.29552f81ff523p-3,
    0x1.371fc201e8f74p-3, 0x1.44d2b6ccb7d1ep-3, 0x1.526e5e3a1b437p-3,
    0x1.5ff3070a793d3p-3, 0x1.6d60fe719d21cp-3, 0x1.7ab890210d909p-3,
    0x1.87fa06520c910p-3, 0x1.9525a9cf456b4p-3, 0x1.a23bc1fe2b563p-3,
    0x1.af3c94e80bff2p-3, 0x1.bc286742d8cd6p-3, 0x1.c8ff7c79a9a21p-3,
    0x1.d5c216b4fbb91p-3, 0x1.e27076e2af2e5p-3, 0x1.ef0adcbdc5936p-3,
    0x1.fb9186d5e3e2ap-3, 0x1.0402594b4d040p-2, 0x1.0a324e27390e3p-2,
    0x1.1058bf9ae4ad5p-2, 0x1.1675cababa60ep-2, 0x1.1c898c16999fap-2,
    0x1.22941fbcf7965p-2, 0x1.2895a13de86a3p-2, 0x1.2e8e2bae11d30p-2,
    0x1.347dd9a987d54p-2, 0x1.3a64c556945e9p-2, 0x1.404308686a7e3p-2,
    0x1.4618bc21c5ec2p-2, 0x1.4be5f957778a0p-2, 0x1.51aad872df82dp-2,
    0x1.5767717455a6cp-2, 0x1.5d1bdbf5809cap-2, 0x1.62c82f2b9c795p-2,
    0x1.686c81e9b14aep-2, 0x1.6e08eaa2ba1e3p-2, 0x1.739d7f6bbd006p-2,
    0x1.792a55fdd47a2p-2, 0x1.7eaf83b82afc3p-2, 0x1.842d1da1e8b17p-2,
    0x1.89a3386c1425ap-2, 0x1.8f11e873662c7p-2, 0x1.947941c2116fap-2,
    0x1.99d958117e08ap-2, 0x1.9f323ecbf984bp-2, 0x1.a484090e5bb0ap-2,
    0x1.a9cec9a9a0849p-2, 0x1.af1293247786bp-2, 0x1.b44f77bcc8f62p-2,
    0x1.b9858969310fbp-2, 0x1.beb4d9da71b7bp-2, 0x1.c3dd7a7cdad4dp-2,
    0x1.c8ff7c79a9a21p-2, 0x1.ce1af0b85f3ebp-2, 0x1.d32fe7e00ebd5p-2,
    0x1.d83e7258a2f3ep-2, 0x1.dd46a04c1c4a0p-2, 0x1.e24881a7c6c26p-2,
    0x1.e744261d68787p-2, 0x1.ec399d2468cc0p-2, 0x1.f128f5faf06ecp-2,
    0x1.f6123fa7028acp-2, 0x1.faf588f78f31ep-2, 0x1.ffd2e0857f498p-2,
    0x1.02552a5a5d0fep-1, 0x1.04bdf9da926d2p-1, 0x1.0723e5c1cdf40p-1,
    0x1.0986f4f573520p-1, 0x1.0be72e4252a82p-1, 0x1.0e44985d1cc8bp-1,
    0x1.109f39e2d4c96p-1, 0x1.12f719593efbcp-1, 0x1.154c3d2f4d5e9p-1,
    0x1.179eabbd899a0p-1, 0x1.19ee6b467c96ep-1, 0x1.1c3b81f713c24p-1,
    0x1.1e85f5e7040d0p-1, 0x1.20cdcd192ab6dp-1, 0x1.23130d7bebf42p-1,
    0x1.2555bce98f7cbp-1, 0x1.2795e1289b11ap-1, 0x1.29d37fec2b08ap-1,
    0x1.2c0e9ed448e8bp-1, 0x1.2e47436e40268p-1, 0x1.307d7334f10bep-1,
    0x1.32b1339121d71p-1, 0x1.34e289d9ce1d3p-1, 0x1.37117b54747b5p-1,
    0x1.393e0d3562a19p-1, 0x1.3b68449fffc22p-1, 0x1.3d9026a7156fap-1,
    0x1.3fb5b84d16f42p-1, 0x1.41d8fe84672aep-1, 0x1.43f9fe2f9ce67p-1,
    0x1.4618bc21c5ec2p-1, 0x1.48353d1ea88dfp-1, 0x1.4a4f85db03ebbp-1,
    0x1.4c679afccee39p-1, 0x1.4e7d811b75bb0p-1, 0x1.50913cc01686bp-1,
    0x1.52a2d265bc5aap-1, 0x1.54b2467999497p-1, 0x1.56bf9d5b3f399p-1,
    0x1.58cadb5cd7989p-1, 0x1.5ad404c359f2cp-1, 0x1.5cdb1dc6c1764p-1,
    0x1.5ee02a9241675p-1, 0x1.60e32f44788d8p-1};

INLINE_FMA
LLVM_LIBC_FUNCTION(float, logf, (float x)) {
  constexpr double LOG_2 = 0x1.62e42fefa39efp-1;
  using FPBits = typename fputil::FPBits<float>;
  FPBits xbits(x);

  switch (FPBits(x).uintval()) {
  case 0x41178febU: // x = 0x1.2f1fd6p+3f
    if (fputil::get_round() == FE_TONEAREST)
      return 0x1.1fcbcep+1f;
    break;
  case 0x4c5d65a5U: // x = 0x1.bacb4ap+25f
    if (fputil::get_round() == FE_TONEAREST)
      return 0x1.1e0696p+4f;
    break;
  case 0x65d890d3U: // x = 0x1.b121a6p+76f
    if (fputil::get_round() == FE_TONEAREST)
      return 0x1.a9a3f2p+5f;
    break;
  case 0x6f31a8ecU: // x = 0x1.6351d8p+95f
    if (fputil::get_round() == FE_TONEAREST)
      return 0x1.08b512p+6f;
    break;
  case 0x3f800001U: // x = 0x1.000002p+0f
    if (fputil::get_round() == FE_UPWARD)
      return 0x1p-23f;
    return 0x1.fffffep-24f;
  case 0x500ffb03U: // x = 0x1.1ff606p+33f
    if (fputil::get_round() != FE_UPWARD)
      return 0x1.6fdd34p+4f;
    break;
  case 0x7a17f30aU: // x = 0x1.2fe614p+117f
    if (fputil::get_round() != FE_UPWARD)
      return 0x1.451436p+6f;
    break;
  case 0x5cd69e88U: // x = 0x1.ad3d1p+58f
    if (fputil::get_round() != FE_UPWARD)
      return 0x1.45c146p+5f;
    break;
  }

  int m = 0;

  if (xbits.uintval() < FPBits::MIN_NORMAL ||
      xbits.uintval() > FPBits::MAX_NORMAL) {
    if (xbits.is_zero()) {
      return static_cast<float>(FPBits::neg_inf());
    }
    if (xbits.get_sign() && !xbits.is_nan()) {
      return FPBits::build_nan(1 << (fputil::MantissaWidth<float>::VALUE - 1));
    }
    if (xbits.is_inf_or_nan()) {
      return x;
    }
    // Normalize denormal inputs.
    xbits.val *= 0x1.0p23f;
    m = -23;
  }

  m += xbits.get_exponent();
  // Set bits to 1.m
  xbits.set_unbiased_exponent(0x7F);
  int f_index = xbits.get_mantissa() >> 16;

  FPBits f(xbits.val);
  f.bits &= ~0x0000'FFFF;

  double d = static_cast<float>(xbits) - static_cast<float>(f);
  d *= ONE_OVER_F[f_index];

  double extra_factor =
      fputil::fma(static_cast<double>(m), LOG_2, LOG_F[f_index]);

  double r = __llvm_libc::fputil::polyeval(
      d, extra_factor, 0x1.fffffffffffacp-1, -0x1.fffffffef9cb2p-2,
      0x1.5555513bc679ap-2, -0x1.fff4805ea441p-3, 0x1.930180dbde91ap-3);

  return static_cast<float>(r);
}

#pragma clang diagnostic pop

} // namespace __llvm_libc
