//===-- Single-precision 2^x function -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/exp2f.h"
#include "src/__support/FPUtil/BasicOperations.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/FMA.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/PolyEval.h"
#include "src/__support/common.h"

#include <errno.h>

namespace __llvm_libc {

// Lookup table for 2^(m * 2^(-7)) with m = 0, ..., 127.
// Table is generated with Sollya as follow:
// > display = hexadecimal;
// > for i from 0 to 127 do { D(2^(i / 128)); };
static constexpr double EXP_M[128] = {
    0x1.0000000000000p0, 0x1.0163da9fb3335p0, 0x1.02c9a3e778061p0,
    0x1.04315e86e7f85p0, 0x1.059b0d3158574p0, 0x1.0706b29ddf6dep0,
    0x1.0874518759bc8p0, 0x1.09e3ecac6f383p0, 0x1.0b5586cf9890fp0,
    0x1.0cc922b7247f7p0, 0x1.0e3ec32d3d1a2p0, 0x1.0fb66affed31bp0,
    0x1.11301d0125b51p0, 0x1.12abdc06c31ccp0, 0x1.1429aaea92de0p0,
    0x1.15a98c8a58e51p0, 0x1.172b83c7d517bp0, 0x1.18af9388c8deap0,
    0x1.1a35beb6fcb75p0, 0x1.1bbe084045cd4p0, 0x1.1d4873168b9aap0,
    0x1.1ed5022fcd91dp0, 0x1.2063b88628cd6p0, 0x1.21f49917ddc96p0,
    0x1.2387a6e756238p0, 0x1.251ce4fb2a63fp0, 0x1.26b4565e27cddp0,
    0x1.284dfe1f56381p0, 0x1.29e9df51fdee1p0, 0x1.2b87fd0dad990p0,
    0x1.2d285a6e4030bp0, 0x1.2ecafa93e2f56p0, 0x1.306fe0a31b715p0,
    0x1.32170fc4cd831p0, 0x1.33c08b26416ffp0, 0x1.356c55f929ff1p0,
    0x1.371a7373aa9cbp0, 0x1.38cae6d05d866p0, 0x1.3a7db34e59ff7p0,
    0x1.3c32dc313a8e5p0, 0x1.3dea64c123422p0, 0x1.3fa4504ac801cp0,
    0x1.4160a21f72e2ap0, 0x1.431f5d950a897p0, 0x1.44e086061892dp0,
    0x1.46a41ed1d0057p0, 0x1.486a2b5c13cd0p0, 0x1.4a32af0d7d3dep0,
    0x1.4bfdad5362a27p0, 0x1.4dcb299fddd0dp0, 0x1.4f9b2769d2ca7p0,
    0x1.516daa2cf6642p0, 0x1.5342b569d4f82p0, 0x1.551a4ca5d920fp0,
    0x1.56f4736b527dap0, 0x1.58d12d497c7fdp0, 0x1.5ab07dd485429p0,
    0x1.5c9268a5946b7p0, 0x1.5e76f15ad2148p0, 0x1.605e1b976dc09p0,
    0x1.6247eb03a5585p0, 0x1.6434634ccc320p0, 0x1.6623882552225p0,
    0x1.68155d44ca973p0, 0x1.6a09e667f3bcdp0, 0x1.6c012750bdabfp0,
    0x1.6dfb23c651a2fp0, 0x1.6ff7df9519484p0, 0x1.71f75e8ec5f74p0,
    0x1.73f9a48a58174p0, 0x1.75feb564267c9p0, 0x1.780694fde5d3fp0,
    0x1.7a11473eb0187p0, 0x1.7c1ed0130c132p0, 0x1.7e2f336cf4e62p0,
    0x1.80427543e1a12p0, 0x1.82589994cce13p0, 0x1.8471a4623c7adp0,
    0x1.868d99b4492edp0, 0x1.88ac7d98a6699p0, 0x1.8ace5422aa0dbp0,
    0x1.8cf3216b5448cp0, 0x1.8f1ae99157736p0, 0x1.9145b0b91ffc6p0,
    0x1.93737b0cdc5e5p0, 0x1.95a44cbc8520fp0, 0x1.97d829fde4e50p0,
    0x1.9a0f170ca07bap0, 0x1.9c49182a3f090p0, 0x1.9e86319e32323p0,
    0x1.a0c667b5de565p0, 0x1.a309bec4a2d33p0, 0x1.a5503b23e255dp0,
    0x1.a799e1330b358p0, 0x1.a9e6b5579fdbfp0, 0x1.ac36bbfd3f37ap0,
    0x1.ae89f995ad3adp0, 0x1.b0e07298db666p0, 0x1.b33a2b84f15fbp0,
    0x1.b59728de5593ap0, 0x1.b7f76f2fb5e47p0, 0x1.ba5b030a1064ap0,
    0x1.bcc1e904bc1d2p0, 0x1.bf2c25bd71e09p0, 0x1.c199bdd85529cp0,
    0x1.c40ab5fffd07ap0, 0x1.c67f12e57d14bp0, 0x1.c8f6d9406e7b5p0,
    0x1.cb720dcef9069p0, 0x1.cdf0b555dc3fap0, 0x1.d072d4a07897cp0,
    0x1.d2f87080d89f2p0, 0x1.d5818dcfba487p0, 0x1.d80e316c98398p0,
    0x1.da9e603db3285p0, 0x1.dd321f301b460p0, 0x1.dfc97337b9b5fp0,
    0x1.e264614f5a129p0, 0x1.e502ee78b3ff6p0, 0x1.e7a51fbc74c83p0,
    0x1.ea4afa2a490dap0, 0x1.ecf482d8e67f1p0, 0x1.efa1bee615a27p0,
    0x1.f252b376bba97p0, 0x1.f50765b6e4540p0, 0x1.f7bfdad9cbe14p0,
    0x1.fa7c1819e90d8p0, 0x1.fd3c22b8f71f1p0,
};

INLINE_FMA
LLVM_LIBC_FUNCTION(float, exp2f, (float x)) {
  using FPBits = typename fputil::FPBits<float>;
  FPBits xbits(x);

  // When x < -150 or nan
  if (unlikely(xbits.uintval() > 0xc316'0000U)) {
    // exp(-Inf) = 0
    if (xbits.is_inf())
      return 0.0f;
    // exp(nan) = nan
    if (xbits.is_nan())
      return x;
    if (fputil::get_round() == FE_UPWARD)
      return static_cast<float>(FPBits(FPBits::MIN_SUBNORMAL));
    if (x != 0.0f)
      errno = ERANGE;
    return 0.0f;
  }
  // x >= 128 or nan
  if (unlikely(!xbits.get_sign() && (xbits.uintval() >= 0x4300'0000U))) {
    if (xbits.uintval() < 0x7f80'0000U) {
      int rounding = fputil::get_round();
      if (rounding == FE_DOWNWARD || rounding == FE_TOWARDZERO)
        return static_cast<float>(FPBits(FPBits::MAX_NORMAL));

      errno = ERANGE;
    }
    return x + static_cast<float>(FPBits::inf());
  }
  // |x| < 2^-25
  if (unlikely(xbits.get_unbiased_exponent() <= 101)) {
    return 1.0f + x;
  }
  // Exceptional values.
  switch (xbits.uintval()) {
  case 0x3b42'9d37U: // x = 0x1.853a6ep-9f
    if (fputil::get_round() == FE_TONEAREST)
      return 0x1.00870ap+0f;
    break;
  case 0xbcf3'a937U: // x = -0x1.e7526ep-6f
    if (fputil::get_round() == FE_TONEAREST)
      return 0x1.f58d62p-1f;
    break;
  }

  // For -150 <= x < 128, to compute 2^x, we perform the following range
  // reduction: find hi, mid, lo such that:
  //   x = hi + mid + lo, in which
  //     hi is an integer,
  //     mid * 2^7 is an integer
  //     -2^(-8) <= lo < 2^-8.
  // In particular,
  //   hi + mid = round(x * 2^7) * 2^(-7).
  // Then,
  //   2^(x) = 2^(hi + mid + lo) = 2^hi * 2^mid * 2^lo.
  // Multiply by 2^hi is simply adding hi to the exponent field.  We store
  // exp(mid) in the lookup tables EXP_M.  exp(lo) is computed using a degree-7
  // minimax polynomial generated by Sollya.

  // x_hi = hi + mid.
  int x_hi = static_cast<int>(x * 0x1.0p7f);
  // Subtract (hi + mid) from x to get lo.
  x -= static_cast<float>(x_hi) * 0x1.0p-7f;
  double xd = static_cast<double>(x);
  // Make sure that -2^(-8) <= lo < 2^-8.
  if (x >= 0x1.0p-8f) {
    ++x_hi;
    xd -= 0x1.0p-7;
  }
  if (x < -0x1.0p-8f) {
    --x_hi;
    xd += 0x1.0p-7;
  }
  // For 2-complement integers, arithmetic right shift is the same as dividing
  // by a power of 2 and then round down (toward negative infinity).
  int hi = x_hi >> 7;
  // mid = x_hi & 0x0000'007fU;
  double exp_mid = EXP_M[x_hi & 0x7f];
  // Degree-6 minimax polynomial generated by Sollya with the following
  // commands:
  //   > display = hexadecimal;
  //   > Q = fpminimax((2^x - 1)/x, 5, [|D...|], [-2^-8, 2^-8]);
  //   > Q;
  double exp_lo =
      fputil::polyeval(xd, 0x1p0, 0x1.62e42fefa39efp-1, 0x1.ebfbdff82c58ep-3,
                       0x1.c6b08d711fe2fp-5, 0x1.3b2ab6fe3deb5p-7,
                       0x1.5d72a05f45c04p-10, 0x1.4284d40c33326p-13);
  fputil::FPBits<double> result(exp_mid * exp_lo);
  result.set_unbiased_exponent(static_cast<uint16_t>(
      static_cast<int>(result.get_unbiased_exponent()) + hi));
  return static_cast<float>(static_cast<double>(result));
}

} // namespace __llvm_libc
