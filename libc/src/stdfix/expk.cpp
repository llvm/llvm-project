//===-- Implementation of expk function ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "expk.h"
#include "src/__support/CPP/bit.h"
#include "src/__support/common.h"
#include "src/__support/fixed_point/fx_bits.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

namespace {

// Look up tables for exp(hi) and exp(mid).
// Generated with Sollya:
// > for i from 0 to 23 do {
//     hi = i - 11;
//     e_hi = nearestint(exp(hi) * 2^15) * 2^-15;
//     print(e_hi, "k,");
//   };
static constexpr accum EXP_HI[24] = {
    0x1p-15k,        0x1p-15k,         0x1p-13k,        0x1.6p-12k,
    0x1.ep-11k,      0x1.44p-9k,       0x1.bap-8k,      0x1.2cp-6k,
    0x1.97cp-5k,     0x1.153p-3k,      0x1.78b8p-2k,    0x1p0k,
    0x1.5bf1p1k,     0x1.d8e68p2k,     0x1.415e6p4k,    0x1.b4c9p5k,
    0x1.28d388p7k,   0x1.936dc6p8k,    0x1.1228858p10k, 0x1.749ea7cp11k,
    0x1.fa7157cp12k, 0x1.5829dcf8p14k, 0x1.d3c4489p15k, ACCUM_MAX,
};

// Generated with Sollya:
// > for i from 0 to 15 do {
//     m = i/16 - 0.0625;
//     e_m = nearestint(exp(m) * 2^15) * 2^-15;
//     print(e_m, "k,");
//   };
static constexpr accum EXP_MID[16] = {
    0x1.e0fcp-1k, 0x1p0k,      0x1.1082p0k, 0x1.2216p0k,
    0x1.34ccp0k,  0x1.48b6p0k, 0x1.5deap0k, 0x1.747ap0k,
    0x1.8c8p0k,   0x1.a612p0k, 0x1.c14cp0k, 0x1.de46p0k,
    0x1.fd1ep0k,  0x1.0efap1k, 0x1.2074p1k, 0x1.330ep1k,
};

} // anonymous namespace

LLVM_LIBC_FUNCTION(accum, expk, (accum x)) {
  using FXRep = fixed_point::FXRep<accum>;
  using StorageType = typename FXRep::StorageType;
  // Output overflow
  // > floor(log(2^16) * 2^15) * 2^-15
  if (LIBC_UNLIKELY(x >= 0x1.62e4p3k))
    return FXRep::MAX();
  // Lower bound where exp(x) -> 0:
  //   floor(log(2^-16) * 2^15) * 2^-15
  if (LIBC_UNLIKELY(x <= -0x1.62e44p3k))
    return FXRep::ZERO();

  // Current range of x:
  //   -0x1.62e4p3 <= x <= 0x1.62e3cp3
  // Range reduction:
  //   x = hi + mid + lo,
  // where:
  //   hi is an integer
  //   mid * 2^4 is an integer
  //   |lo| <= 2^-5.
  // Then exp(x) = exp(hi + mid + lo) = exp(hi) * exp(mid) * exp(lo)
  //             ~ exp(hi) * exp(mid) * (1 + lo + lo^2 / 2)
  // with relative errors < |lo|^3/2 <= 2^-16.
  //   exp(hi) and exp(mid) are extracted from small lookup tables.

  // Round-to-nearest 1/16, tie-to-(+Int):
  constexpr accum ONE_THIRTY_SECOND = 0x1.0p-5k;
  // x_rounded = floor(x + 1/16).
  accum x_rounded = ((x + ONE_THIRTY_SECOND) >> (FXRep::FRACTION_LEN - 4))
                    << (FXRep::FRACTION_LEN - 4);
  accum lo = x - x_rounded;

  // Range of x_rounded:
  //   x_rounded >= floor((-0x1.62e4p3 + 0x1.0p-5) * 2^4) * 2^-4
  //              = -0x1.62p3 = -11.0625
  // To get the indices, we shift the values so that it start with 0.
  // Range of indices: 0 <= indices <= 355.
  StorageType indices = cpp::bit_cast<StorageType>((x_rounded + 0x1.62p3k) >>
                                                   (FXRep::FRACTION_LEN - 4));
  // So we have the following relation:
  //   indices = (hi + mid + 177/16) * 16
  // That implies:
  //   hi + mid = indices/16 - 11.0625
  // So for lookup tables, we can use the upper 4 bits to get:
  //   exp( floor(indices / 16) - 11 )
  // and lower 4 bits for:
  //   exp( (indices - floor(indices)) - 0.0625 )
  accum exp_hi = EXP_HI[indices >> 4];
  accum exp_mid = EXP_MID[indices & 0xf];
  // exp(x) ~ exp(hi) * exp(mid) * (1 + lo);
  accum l1 = 0x1.0p0k + (lo >> 1); // = 1 + lo / 2
  accum l2 = 0x1.0p0k + lo * l1;   // = 1 + lo * (1 + lo / 2) = 1 + lo + lo^2/2
  return (exp_hi * (exp_mid * l2));
}

} // namespace LIBC_NAMESPACE_DECL
