//===-- Single-precision log10(x) function --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/log10f.h"
#include "common_constants.h" // Lookup table for (1/f)
#include "src/__support/FPUtil/BasicOperations.h"
#include "src/__support/FPUtil/FEnvUtils.h"
#include "src/__support/FPUtil/FMA.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/PolyEval.h"
#include "src/__support/common.h"

// This is an algorithm for log10(x) in single precision which is
// correctly rounded for all rounding modes, based on the implementation of
// log10(x) from the RLIBM project at:
// https://people.cs.rutgers.edu/~sn349/rlibm

// Step 1 - Range reduction:
//   For x = 2^m * 1.mant, log(x) = m * log10(2) + log10(1.m)
//   If x is denormal, we normalize it by multiplying x by 2^23 and subtracting
//   m by 23.

// Step 2 - Another range reduction:
//   To compute log(1.mant), let f be the highest 8 bits including the hidden
// bit, and d be the difference (1.mant - f), i.e. the remaining 16 bits of the
// mantissa. Then we have the following approximation formula:
//   log10(1.mant) = log10(f) + log10(1.mant / f)
//                 = log10(f) + log10(1 + d/f)
//                 ~ log10(f) + P(d/f)
// since d/f is sufficiently small.
//   log10(f) and 1/f are then stored in two 2^7 = 128 entries look-up tables.

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

// Exact power of 10 in float:

// Lookup table for log10(f) = log10(1 + n*2^(-7)) where n = 0..127.
static constexpr double LOG10_F[128] = {
    0x0.0000000000000p+0, 0x1.bafd47221ed26p-9, 0x1.b9476a4fcd10fp-8,
    0x1.49b0851443684p-7, 0x1.b5e908eb13790p-7, 0x1.10a83a8446c78p-6,
    0x1.45f4f5acb8be0p-6, 0x1.7adc3df3b1ff8p-6, 0x1.af5f92b00e610p-6,
    0x1.e3806acbd058fp-6, 0x1.0ba01a8170000p-5, 0x1.25502c0fc314cp-5,
    0x1.3ed1199a5e425p-5, 0x1.58238eeb353dap-5, 0x1.71483427d2a99p-5,
    0x1.8a3fadeb847f4p-5, 0x1.a30a9d609efeap-5, 0x1.bba9a058dfd84p-5,
    0x1.d41d5164facb4p-5, 0x1.ec6647eb58808p-5, 0x1.02428c1f08016p-4,
    0x1.0e3d29d81165ep-4, 0x1.1a23445501816p-4, 0x1.25f5215eb594ap-4,
    0x1.31b3055c47118p-4, 0x1.3d5d335c53179p-4, 0x1.48f3ed1df48fbp-4,
    0x1.5477731973e85p-4, 0x1.5fe80488af4fdp-4, 0x1.6b45df6f3e2c9p-4,
    0x1.769140a2526fdp-4, 0x1.81ca63d05a44ap-4, 0x1.8cf183886480dp-4,
    0x1.9806d9414a209p-4, 0x1.a30a9d609efeap-4, 0x1.adfd07416be07p-4,
    0x1.b8de4d3ab3d98p-4, 0x1.c3aea4a5c6effp-4, 0x1.ce6e41e463da5p-4,
    0x1.d91d5866aa99cp-4, 0x1.e3bc1ab0e19fep-4, 0x1.ee4aba610f204p-4,
    0x1.f8c9683468191p-4, 0x1.019c2a064b486p-3, 0x1.06cbd67a6c3b6p-3,
    0x1.0bf3d0937c41cp-3, 0x1.11142f0811357p-3, 0x1.162d082ac9d10p-3,
    0x1.1b3e71ec94f7bp-3, 0x1.204881dee8777p-3, 0x1.254b4d35e7d3cp-3,
    0x1.2a46e8ca7ba2ap-3, 0x1.2f3b691c5a001p-3, 0x1.3428e2540096dp-3,
    0x1.390f6844a0b83p-3, 0x1.3def0e6dfdf85p-3, 0x1.42c7e7fe3fc02p-3,
    0x1.479a07d3b6411p-3, 0x1.4c65807e93338p-3, 0x1.512a644296c3dp-3,
    0x1.55e8c518b10f8p-3, 0x1.5aa0b4b0988fap-3, 0x1.5f52447255c92p-3,
    0x1.63fd857fc49bbp-3, 0x1.68a288b60b7fcp-3, 0x1.6d415eaf0906bp-3,
    0x1.71da17c2b7e80p-3, 0x1.766cc40889e85p-3, 0x1.7af97358b9e04p-3,
    0x1.7f80354d952a0p-3, 0x1.84011944bcb75p-3, 0x1.887c2e605e119p-3,
    0x1.8cf183886480dp-3, 0x1.9161276ba2978p-3, 0x1.95cb2880f45bap-3,
    0x1.9a2f95085a45cp-3, 0x1.9e8e7b0c0d4bep-3, 0x1.a2e7e8618c2d2p-3,
    0x1.a73beaaaa22f4p-3, 0x1.ab8a8f56677fcp-3, 0x1.afd3e3a23b680p-3,
    0x1.b417f49ab8807p-3, 0x1.b856cf1ca3105p-3, 0x1.bc907fd5d1c40p-3,
    0x1.c0c5134610e26p-3, 0x1.c4f495c0002a2p-3, 0x1.c91f1369eb7cap-3,
    0x1.cd44983e9e7bdp-3, 0x1.d165300e333f7p-3, 0x1.d580e67edc43dp-3,
    0x1.d997c70da9b47p-3, 0x1.dda9dd0f4a329p-3, 0x1.e1b733b0c7381p-3,
    0x1.e5bfd5f83d342p-3, 0x1.e9c3cec58f807p-3, 0x1.edc328d3184afp-3,
    0x1.f1bdeeb654901p-3, 0x1.f5b42ae08c407p-3, 0x1.f9a5e79f76ac5p-3,
    0x1.fd932f1ddb4d6p-3, 0x1.00be05b217844p-2, 0x1.02b0432c96ff0p-2,
    0x1.04a054e139004p-2, 0x1.068e3fa282e3dp-2, 0x1.087a0832fa7acp-2,
    0x1.0a63b3456c819p-2, 0x1.0c4b457d3193dp-2, 0x1.0e30c36e71a7fp-2,
    0x1.1014319e661bdp-2, 0x1.11f594839a5bdp-2, 0x1.13d4f0862b2e1p-2,
    0x1.15b24a0004a92p-2, 0x1.178da53d1ee01p-2, 0x1.1967067bb94b8p-2,
    0x1.1b3e71ec94f7bp-2, 0x1.1d13ebb32d7f9p-2, 0x1.1ee777e5f0dc3p-2,
    0x1.20b91a8e76105p-2, 0x1.2288d7a9b2b64p-2, 0x1.2456b3282f786p-2,
    0x1.2622b0ee3b79dp-2, 0x1.27ecd4d41eb67p-2, 0x1.29b522a64b609p-2,
    0x1.2b7b9e258e422p-2, 0x1.2d404b073e27ep-2, 0x1.2f032cf56a5bep-2,
    0x1.30c4478f0835fp-2, 0x1.32839e681fc62p-2};

INLINE_FMA
LLVM_LIBC_FUNCTION(float, log10f, (float x)) {
  constexpr double LOG10_2 = 0x1.34413509f79ffp-2;

  using FPBits = typename fputil::FPBits<float>;
  FPBits xbits(x);
  double m = 0.0;

  // Exact powers of 10 and other hard-to-round cases.
  switch (xbits.uintval()) {
  case 0x4120'0000U: // x = 10
    return 1.0f;
  case 0x42c8'0000U: // x = 100
    return 2.0f;
  case 0x447a'0000U: // x = 1,000
    return 3.0f;
  case 0x461c'4000U: // x = 10,000
    return 4.0f;
  case 0x47c3'5000U: // x = 100,000
    return 5.0f;
  case 0x4974'2400U: // x = 1,000,000
    return 6.0f;
  case 0x4b18'9680U: // x = 10,000,000
    return 7.0f;
  case 0x4cbe'bc20U: // x = 100,000,000
    return 8.0f;
  case 0x4e6e'6b28U: // x = 1,000,000,000
    return 9.0f;
  case 0x5015'02f9U: // x = 10,000,000,000
    return 10.0f;
  case 0x4f13'4f83U: // x = 2471461632.0
    if (fputil::get_round() == FE_UPWARD)
      return 0x1.2c9314p+3f;
    break;
  case 0x7956'ba5eU: { // x = 69683218960000541503257137270226944.0
    int round_mode = fputil::get_round();
    if (round_mode == FE_DOWNWARD || round_mode == FE_TOWARDZERO)
      return 0x1.16bebap+5f;
    break;
  }
  }

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
    xbits.set_val(xbits.get_val() * 0x1.0p23f);
    m -= 23.0;
  }

  m += static_cast<double>(xbits.get_exponent());
  // Set bits to 1.m
  xbits.set_unbiased_exponent(0x7F);
  int f_index = xbits.get_mantissa() >> 16;

  FPBits f = xbits;
  f.bits &= ~0x0000'FFFF;

  double d = static_cast<float>(xbits) - static_cast<float>(f);
  d *= ONE_OVER_F[f_index];

  double extra_factor = fputil::fma(m, LOG10_2, LOG10_F[f_index]);

  double r = fputil::polyeval(d, extra_factor, 0x1.bcb7b1526e4c5p-2,
                              -0x1.bcb7b1518a5e9p-3, 0x1.287a72a6f716p-3,
                              -0x1.bcadb40b85565p-4, 0x1.5e0bc97f97e22p-4);

  return static_cast<float>(r);
}

} // namespace __llvm_libc
