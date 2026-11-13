//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of expuhk function.
///
//===----------------------------------------------------------------------===//

#include "src/stdfix/expuhk.h"
#include "src/__support/CPP/bit.h"
#include "src/__support/common.h"
#include "src/__support/fixed_point/fx_rep.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/optimization.h"

namespace LIBC_NAMESPACE_DECL {

namespace {

// Look up tables for exp(hi) and exp(mid).
// Generated with Sollya:
// > for hi from 0 to 5 do {
//     e_hi = nearestint(exp(hi) * 2^8) * 2^-8;
//     print(hi, e_hi);
//   };
// > for i from 0 to 7 do {
//     m = i/8;
//     e_mid = nearestint(exp(m) * 2^8) * 2^-8;
//     print(m, e_mid);
//   };
static constexpr unsigned short accum EXP_HI[6] = {
    0x1.0p0uhk,   0x1.5cp1uhk,   0x1.d9p2uhk,
    0x1.416p4uhk, 0x1.b4c8p5uhk, 0x1.28d4p7uhk,
};

static constexpr unsigned short accum EXP_MID[8] = {
    0x1.0p0uhk,  0x1.22p0uhk, 0x1.49p0uhk, 0x1.74p0uhk,
    0x1.a6p0uhk, 0x1.dep0uhk, 0x1.0fp1uhk, 0x1.33p1uhk,
};

} // anonymous namespace

LLVM_LIBC_FUNCTION(unsigned short accum, expuhk, (unsigned short accum x)) {
  using FXRep = fixed_point::FXRep<unsigned short accum>;
  using StorageType = typename FXRep::StorageType;

  // Output overflow
  //   ceil(log(2^8) * 2^8) * 2^-8
  if (LIBC_UNLIKELY(x >= 0x1.63p2uhk))
    return FXRep::MAX();

  // Current range of x:
  //   0 <= x <= 0x1.62cp2
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

  // Round-to-nearest 1/8, tie-to-(+Inf):
  constexpr unsigned short accum ONE_SIXTEENTH = 0x1.0p-4uhk;
  // x_rounded = floor((x + 1/16) * 2^3) * 2^-3.
  unsigned short accum x_rounded =
      ((x + ONE_SIXTEENTH) >> (FXRep::FRACTION_LEN - 3))
      << (FXRep::FRACTION_LEN - 3);
  unsigned short accum one_plus_lo = (0x1.0p0uhk + x) - x_rounded; // = 1 + lo

  // Range of x_rounded:
  //   0 <= x_rounded <= floor((0x1.62cp2 + 0x1.0p-4) * 2^3) * 2^-3
  //                   = 0x1.6p2 = 5.5
  // Range of indices:  0 <= indices <= 44
  StorageType indices =
      cpp::bit_cast<StorageType>(x_rounded >> (FXRep::FRACTION_LEN - 3));
  // So we have the following relation:
  //   indices = (hi + mid) * 8
  // That implies:
  //   hi + mid = indices/8
  // So for lookup tables, we can use the upper 3 bits to get:
  //   exp( floor(indices / 8) )
  // and lower 3 bits for:
  //   exp( indices/8 - floor(indices / 8) )
  unsigned short accum exp_hi = EXP_HI[indices >> 3];
  unsigned short accum exp_mid = EXP_MID[indices & 0x7];
  // exp(x) ~ exp(hi) * exp(mid) * (1 + lo)
  return exp_hi * (exp_mid * one_plus_lo);
}

} // namespace LIBC_NAMESPACE_DECL
