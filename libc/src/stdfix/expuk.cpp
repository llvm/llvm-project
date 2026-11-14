//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of expuk function.
///
//===----------------------------------------------------------------------===//

#include "src/stdfix/expuk.h"
#include "src/__support/CPP/bit.h"
#include "src/__support/common.h"
#include "src/__support/fixed_point/fx_rep.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/optimization.h"

namespace LIBC_NAMESPACE_DECL {

namespace {

// Look up tables for exp(hi) and exp(mid).
// Generated with Sollya:
// > for hi from 0 to 11 do {
//     e_hi = nearestint(exp(hi) * 2^16) * 2^-16;
//     print(e_hi, "uk,");
//   };
static constexpr unsigned accum EXP_HI[12] = {
    0x1.0p0uk,        0x1.5bf08p1uk,    0x1.d8e64p2uk,     0x1.415e6p4uk,
    0x1.b4c9p5uk,     0x1.28d38ap7uk,   0x1.936dc5p8uk,    0x1.122885cp10uk,
    0x1.749ea7ep11uk, 0x1.fa7157cp12uk, 0x1.5829dcf8p14uk, 0x1.d3c4488ep15uk,
};

// Generated with Sollya:
// > for i from 0 to 15 do {
//     m = i/16;
//     e_m = nearestint(exp(m) * 2^16) * 2^-16;
//     print(e_m, "uk,");
//   };
static constexpr unsigned accum EXP_MID[16] = {
    0x1.0p0uk,    0x1.1083p0uk, 0x1.2216p0uk,  0x1.34ccp0uk,
    0x1.48b6p0uk, 0x1.5de9p0uk, 0x1.747ap0uk,  0x1.8c8p0uk,
    0x1.a613p0uk, 0x1.c14bp0uk, 0x1.de45p0uk,  0x1.fd1ep0uk,
    0x1.0efap1uk, 0x1.2074p1uk, 0x1.330e8p1uk, 0x1.46dcp1uk,
};

} // anonymous namespace

LLVM_LIBC_FUNCTION(unsigned accum, expuk, (unsigned accum x)) {
  using FXRep = fixed_point::FXRep<unsigned accum>;
  using StorageType = typename FXRep::StorageType;

  // Output overflow
  //   ceil(log(2^16) * 2^16) * 2^-16
  if (LIBC_UNLIKELY(x >= 0x1.62e44p3uk))
    return FXRep::MAX();

  // Current range of x:
  //   0 <= x <= 0x1.62e42p3
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

  // Round-to-nearest 1/16, tie-to-(+Inf):
  constexpr unsigned accum ONE_THIRTY_SECOND = 0x1.0p-5uk;
  // x_rounded = floor((x + 1/32) * 2^4) * 2^-4.
  unsigned accum x_rounded =
      ((x + ONE_THIRTY_SECOND) >> (FXRep::FRACTION_LEN - 4))
      << (FXRep::FRACTION_LEN - 4);
  unsigned accum one_plus_lo = (0x1.0p0uk + x) - x_rounded; // = 1 + lo

  // Range of x_rounded:
  //   0 <= x_rounded <= floor((0x1.62e42p3 + 0x1.0p-5) * 2^4) * 2^-4
  //                   = 0x1.62p3 = 11.0625
  // Range of indices: 0 <= indices <= 177.
  StorageType indices =
      cpp::bit_cast<StorageType>(x_rounded >> (FXRep::FRACTION_LEN - 4));
  // So we have the following relation:
  //   indices = (hi + mid) * 16
  // That implies:
  //   hi + mid = indices/16
  // So for lookup tables, we can use the upper 4 bits to get:
  //   exp( floor(indices / 16) )
  // and lower 4 bits for:
  //   exp( indices/16 - floor(indices / 16) )
  unsigned accum exp_hi = EXP_HI[indices >> 4];
  unsigned accum exp_mid = EXP_MID[indices & 0xf];
  // exp(x) ~ exp(hi) * exp(mid) * (1 + lo + lo^2 / 2)
  // 1 + lo + lo^2 / 2 = (1 + (1 + lo)^2) / 2
  unsigned accum exp_lo = (0x1.0p0uk + one_plus_lo * one_plus_lo) >> 1;
  return exp_hi * (exp_mid * exp_lo);
}

} // namespace LIBC_NAMESPACE_DECL
