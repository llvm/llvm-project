//===-- include/flang/Evaluate/rounding-bits.h ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_EVALUATE_ROUNDING_BITS_H_
#define FORTRAN_EVALUATE_ROUNDING_BITS_H_

#include "flang/Evaluate/target.h"

// A helper class used by Real<> to determine rounding of rational results
// to floating-point values.  Bits lost from intermediate computations by
// being shifted rightward are accumulated in instances of this class.

namespace Fortran::evaluate::value {

class RoundingBits {
public:
  constexpr RoundingBits(
      bool guard = false, bool round = false, bool sticky = false)
      : guard_{guard}, round_{round}, sticky_{sticky} {}

  template <typename FRACTION>
  constexpr RoundingBits(const FRACTION &fraction, int rshift) {
    if (rshift > 0 && rshift < fraction.bits + 1) {
      guard_ = fraction.BTEST(rshift - 1);
    }
    if (rshift > 1 && rshift < fraction.bits + 2) {
      round_ = fraction.BTEST(rshift - 2);
    }
    if (rshift > 2) {
      if (rshift >= fraction.bits + 2) {
        sticky_ = !fraction.IsZero();
      } else {
        auto mask{fraction.MASKR(rshift - 2)};
        sticky_ = !fraction.IAND(mask).IsZero();
      }
    }
  }

  constexpr bool guard() const { return guard_; }
  constexpr bool round() const { return round_; }
  constexpr bool sticky() const { return sticky_; }
  constexpr bool empty() const { return !(guard_ | round_ | sticky_); }

  constexpr bool Negate() {
    bool carry{!sticky_};
    if (carry) {
      carry = !round_;
    } else {
      round_ = !round_;
    }
    if (carry) {
      carry = !guard_;
    } else {
      guard_ = !guard_;
    }
    return carry;
  }

  constexpr bool ShiftLeft() {
    bool oldGuard{guard_};
    guard_ = round_;
    round_ = sticky_;
    return oldGuard;
  }

  constexpr void ShiftRight(bool newGuard) {
    sticky_ |= round_;
    round_ = guard_;
    guard_ = newGuard;
  }

  // Determines whether a value should be rounded by increasing its
  // fraction, given a rounding mode and a summary of the lost bits.
  constexpr bool MustRound(
      Rounding rounding, bool isNegative, bool isOdd) const {
    bool round{false}; // to dodge bogus g++ warning about missing return
    switch (rounding.mode) {
    case common::RoundingMode::TiesToEven:
      round = guard_ && (round_ | sticky_ | isOdd);
      break;
    case common::RoundingMode::ToZero:
      break;
    case common::RoundingMode::Down:
      round = isNegative && !empty();
      break;
    case common::RoundingMode::Up:
      round = !isNegative && !empty();
      break;
    case common::RoundingMode::TiesAwayFromZero:
      round = guard_;
      break;
    }
    return round;
  }

private:
  bool guard_{false}; // 0.5 * ulp (unit in lowest place)
  bool round_{false}; // 0.25 * ulp
  bool sticky_{false}; // true if any lesser-valued bit would be set
};
} // namespace Fortran::evaluate::value
#endif // FORTRAN_EVALUATE_ROUNDING_BITS_H_
