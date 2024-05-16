//===-- Implementation of exphk function ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "exphk.h"
#include "src/__support/CPP/bit.h"
#include "src/__support/common.h"
#include "src/__support/fixed_point/fx_bits.h"

namespace LIBC_NAMESPACE {

namespace {

// Look up tables for exp(hi) and exp(mid).
// Generated with Sollya:
// > for i from 0 to 89 do {
//     hi = floor(i/8) - 5;
//     m = i/8 - floor(i/8) - 0.5;
//     e_hi = nearestint(exp(hi) * 2^7) * 2^-7;
//     e_mid = nearestint(exp(m) * 2^7) * 2^-7;
//     print(hi, e_hi, m, e_mid);
//   };
// Notice that when i = 88 and 89, e_hi will overflow short accum range.
static constexpr short accum EXP_HI[12] = {
    0x1.0p-7hk, 0x1.0p-6hk, 0x1.8p-5hk,  0x1.1p-3hk,  0x1.78p-2hk,  0x1.0p0hk,
    0x1.5cp1hk, 0x1.d9p2hk, 0x1.416p4hk, 0x1.b4dp5hk, 0x1.28d4p7hk, SACCUM_MAX,
};

static constexpr short accum EXP_MID[8] = {
    0x1.38p-1hk, 0x1.6p-1hk, 0x1.9p-1hk, 0x1.c4p-1hk,
    0x1.0p0hk,   0x1.22p0hk, 0x1.48p0hk, 0x1.74p0hk,
};

} // anonymous namespace

LLVM_LIBC_FUNCTION(short accum, exphk, (short accum x)) {
  using FXRep = fixed_point::FXRep<short accum>;
  using StorageType = typename FXRep::StorageType;
  // Output overflow
  if (LIBC_UNLIKELY(x >= 0x1.64p2hk))
    return FXRep::MAX();
  // Lower bound where exp(x) -> 0:
  //   floor(log(2^-8) * 2^7) * 2^-7
  if (LIBC_UNLIKELY(x <= -0x1.63p2hk))
    return FXRep::ZERO();

  // Current range of x:
  //   -0x1.628p2 <= x <= 0x1.638p2
  // Range reduction:
  //   x = hi + mid + lo,
  // where:
  //   hi is an integer
  //   mid * 2^3 is an integer
  //   |lo| <= 2^-4.
  // Then exp(x) = exp(hi + mid + lo) = exp(hi) * exp(mid) * exp(lo)
  //             ~ exp(hi) * exp(mid) * (1 + lo)
  // with relative errors < |lo|^2 <= 2^-8.
  //   exp(hi) and exp(mid) are extracted from small lookup tables.

  // Round-to-nearest 1/8, tie-to-(+Int):
  constexpr short accum ONE_SIXTEENTH = 0x1.0p-4hk;
  // x_rounded = floor(x + 1/16).
  short accum x_rounded = ((x + ONE_SIXTEENTH) >> (FXRep::FRACTION_LEN - 3))
                          << (FXRep::FRACTION_LEN - 3);
  short accum lo = x - x_rounded;

  // Range of x_rounded:
  //   x_rounded >= floor((-0x1.628p2 + 0x1.0p-4) * 2^3) * 2^-3
  //              = -0x1.6p2 = -5.5
  // To get the indices, we shift the values so that it start with 0.
  // Range of indices:  0 <= indices <= 89
  StorageType indices = cpp::bit_cast<StorageType>((x_rounded + 0x1.6p2hk) >>
                                                   (FXRep::FRACTION_LEN - 3));
  // So we have the following relation:
  //   indices = (hi + mid + 44/8) * 8
  // That implies:
  //   hi + mid = indices/8 - 5.5
  // So for lookup tables, we can use the upper 4 bits to get:
  //   exp( floor(indices / 8) - 5 )
  // and lower 3 bits for:
  //   exp( (indices - floor(indices)) - 0.5 )
  short accum exp_hi = EXP_HI[indices >> 3];
  short accum exp_mid = EXP_MID[indices & 0x7];
  // exp(x) ~ exp(hi) * exp(mid) * (1 + lo);
  return (exp_hi * (exp_mid * (0x1.0p0hk + lo)));
}

} // namespace LIBC_NAMESPACE
