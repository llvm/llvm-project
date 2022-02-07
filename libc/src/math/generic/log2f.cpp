//===-- Single-precision log2(x) function ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/log2f.h"
#include "common_constants.h" // Lookup table for (1/f)
#include "src/__support/FPUtil/BasicOperations.h"
#include "src/__support/FPUtil/FEnvUtils.h"
#include "src/__support/FPUtil/FMA.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/PolyEval.h"
#include "src/__support/common.h"

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
static constexpr double LOG2_F[128] = {
    0x0.0000000000000p+0, 0x1.6fe50b6ef0851p-7, 0x1.6e79685c2d22ap-6,
    0x1.11cd1d5133413p-5, 0x1.6bad3758efd87p-5, 0x1.c4dfab90aab5fp-5,
    0x1.0eb389fa29f9bp-4, 0x1.3aa2fdd27f1c3p-4, 0x1.663f6fac91316p-4,
    0x1.918a16e46335bp-4, 0x1.bc84240adabbap-4, 0x1.e72ec117fa5b2p-4,
    0x1.08c588cda79e4p-3, 0x1.1dcd197552b7bp-3, 0x1.32ae9e278ae1ap-3,
    0x1.476a9f983f74dp-3, 0x1.5c01a39fbd688p-3, 0x1.70742d4ef027fp-3,
    0x1.84c2bd02f03b3p-3, 0x1.98edd077e70dfp-3, 0x1.acf5e2db4ec94p-3,
    0x1.c0db6cdd94deep-3, 0x1.d49ee4c325970p-3, 0x1.e840be74e6a4dp-3,
    0x1.fbc16b902680ap-3, 0x1.0790adbb03009p-2, 0x1.11307dad30b76p-2,
    0x1.1ac05b291f070p-2, 0x1.24407ab0e073ap-2, 0x1.2db10fc4d9aafp-2,
    0x1.37124cea4cdedp-2, 0x1.406463b1b0449p-2, 0x1.49a784bcd1b8bp-2,
    0x1.52dbdfc4c96b3p-2, 0x1.5c01a39fbd688p-2, 0x1.6518fe4677ba7p-2,
    0x1.6e221cd9d0cdep-2, 0x1.771d2ba7efb3cp-2, 0x1.800a563161c54p-2,
    0x1.88e9c72e0b226p-2, 0x1.91bba891f1709p-2, 0x1.9a802391e232fp-2,
    0x1.a33760a7f6051p-2, 0x1.abe18797f1f49p-2, 0x1.b47ebf73882a1p-2,
    0x1.bd0f2e9e79031p-2, 0x1.c592fad295b56p-2, 0x1.ce0a4923a587dp-2,
    0x1.d6753e032ea0fp-2, 0x1.ded3fd442364cp-2, 0x1.e726aa1e754d2p-2,
    0x1.ef6d67328e220p-2, 0x1.f7a8568cb06cfp-2, 0x1.ffd799a83ff9bp-2,
    0x1.03fda8b97997fp-1, 0x1.0809cf27f703dp-1, 0x1.0c10500d63aa6p-1,
    0x1.10113b153c8eap-1, 0x1.140c9faa1e544p-1, 0x1.18028cf72976ap-1,
    0x1.1bf311e95d00ep-1, 0x1.1fde3d30e8126p-1, 0x1.23c41d42727c8p-1,
    0x1.27a4c0585cbf8p-1, 0x1.2b803473f7ad1p-1, 0x1.2f56875eb3f26p-1,
    0x1.3327c6ab49ca7p-1, 0x1.36f3ffb6d9162p-1, 0x1.3abb3faa02167p-1,
    0x1.3e7d9379f7016p-1, 0x1.423b07e986aa9p-1, 0x1.45f3a98a20739p-1,
    0x1.49a784bcd1b8bp-1, 0x1.4d56a5b33cec4p-1, 0x1.510118708a8f9p-1,
    0x1.54a6e8ca5438ep-1, 0x1.5848226989d34p-1, 0x1.5be4d0cb51435p-1,
    0x1.5f7cff41e09afp-1, 0x1.6310b8f553048p-1, 0x1.66a008e4788ccp-1,
    0x1.6a2af9e5a0f0ap-1, 0x1.6db196a76194ap-1, 0x1.7133e9b156c7cp-1,
    0x1.74b1fd64e0754p-1, 0x1.782bdbfdda657p-1, 0x1.7ba18f93502e4p-1,
    0x1.7f1322182cf16p-1, 0x1.82809d5be7073p-1, 0x1.85ea0b0b27b26p-1,
    0x1.894f74b06ef8bp-1, 0x1.8cb0e3b4b3bbep-1, 0x1.900e6160002cdp-1,
    0x1.9367f6da0ab2fp-1, 0x1.96bdad2acb5f6p-1, 0x1.9a0f8d3b0e050p-1,
    0x1.9d5d9fd5010b3p-1, 0x1.a0a7eda4c112dp-1, 0x1.a3ee7f38e181fp-1,
    0x1.a7315d02f20c8p-1, 0x1.aa708f58014d3p-1, 0x1.adac1e711c833p-1,
    0x1.b0e4126bcc86cp-1, 0x1.b418734a9008cp-1, 0x1.b74948f5532dap-1,
    0x1.ba769b39e4964p-1, 0x1.bda071cc67e6ep-1, 0x1.c0c6d447c5dd3p-1,
    0x1.c3e9ca2e1a055p-1, 0x1.c7095ae91e1c7p-1, 0x1.ca258dca93316p-1,
    0x1.cd3e6a0ca8907p-1, 0x1.d053f6d260896p-1, 0x1.d3663b27f31d5p-1,
    0x1.d6753e032ea0fp-1, 0x1.d9810643d6615p-1, 0x1.dc899ab3ff56cp-1,
    0x1.df8f02086af2cp-1, 0x1.e29142e0e0140p-1, 0x1.e59063c8822cep-1,
    0x1.e88c6b3626a73p-1, 0x1.eb855f8ca88fbp-1, 0x1.ee7b471b3a950p-1,
    0x1.f16e281db7630p-1, 0x1.f45e08bcf0655p-1, 0x1.f74aef0efafaep-1,
    0x1.fa34e1177c233p-1, 0x1.fd1be4c7f2af9p-1};

INLINE_FMA
LLVM_LIBC_FUNCTION(float, log2f, (float x)) {
  using FPBits = typename fputil::FPBits<float>;
  FPBits xbits(x);
  int m = 0;

  // Hard to round value(s).
  switch (FPBits(x).uintval()) {
  case 0x3f81d0b5U: {
    int rounding_mode = fputil::get_round();
    if (rounding_mode == FE_DOWNWARD || rounding_mode == FE_TOWARDZERO) {
      return 0x1.4cdc4cp-6f;
    }
    break;
  }
  case 0x3f7e3274U:
    if (fputil::get_round() == FE_TONEAREST) {
      return -0x1.4e1d16p-7f;
    }
    break;
  case 0x3f7d57f5U:
    if (fputil::get_round() == FE_TOWARDZERO) {
      return -0x1.ed1c32p-7f;
    }
    break;
  }

  // Exceptional inputs.
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
    m = -23;
  }

  m += xbits.get_exponent();
  // Set bits to 1.m
  xbits.set_unbiased_exponent(0x7F);
  // Get the 8 highest bits, use 7 bits (excluding the implicit hidden bit) for
  // lookup tables.
  int f_index = xbits.get_mantissa() >> 16;

  FPBits f = xbits;
  // Clear the lowest 16 bits.
  f.bits &= ~0x0000'FFFF;

  double d = static_cast<float>(xbits) - static_cast<float>(f);
  d *= ONE_OVER_F[f_index];

  double extra_factor = static_cast<double>(m) + LOG2_F[f_index];
  double r = __llvm_libc::fputil::polyeval(
      d, extra_factor, 0x1.71547652bd4fp+0, -0x1.7154769b978c7p-1,
      0x1.ec71a99e349c8p-2, -0x1.720d90e6aac6cp-2, 0x1.5132da3583dap-2);

  return static_cast<float>(r);
}

} // namespace __llvm_libc
