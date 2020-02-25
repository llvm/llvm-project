//===-- include/flang/Common/unsigned-const-division.h ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_COMMON_UNSIGNED_CONST_DIVISION_H_
#define FORTRAN_COMMON_UNSIGNED_CONST_DIVISION_H_

// Work around unoptimized implementations of unsigned integer division
// by constant values in some compilers (looking at YOU, clang 7!) by
// explicitly implementing integer division by constant divisors as
// multiplication by a fixed-point reciprocal and a right shift.

#include "bit-population-count.h"
#include "leading-zero-bit-count.h"
#include "uint128.h"
#include <cinttypes>
#include <type_traits>

namespace Fortran::common {

template<typename UINT> class FixedPointReciprocal {
public:
  using type = UINT;

private:
  static_assert(std::is_unsigned_v<type>);
  static const int bits{static_cast<int>(8 * sizeof(type))};
  static_assert(bits <= 64);
  using Big = HostUnsignedIntType<bits * 2>;

public:
  static constexpr FixedPointReciprocal For(type n) {
    if (n == 0) {
      return {0, 0};
    } else if ((n & (n - 1)) == 0) {  // n is a power of two
      return {TrailingZeroBitCount(n), 1};
    } else {
      int shift{bits - 1 + BitsNeededFor(n)};
      return {shift, static_cast<type>(((Big{1} << shift) + n - 1) / n)};
    }
  }

  constexpr type Divide(type n) const {
    return static_cast<type>((static_cast<Big>(reciprocal_) * n) >> shift_);
  }

private:
  constexpr FixedPointReciprocal(int s, type r) : shift_{s}, reciprocal_{r} {}

  int shift_;
  type reciprocal_;
};

static_assert(FixedPointReciprocal<std::uint32_t>::For(5).Divide(2000000000u) ==
    400000000u);
static_assert(FixedPointReciprocal<std::uint64_t>::For(10).Divide(
                  10000000000000000u) == 1000000000000000u);

template<typename UINT, std::uint64_t DENOM>
inline constexpr UINT DivideUnsignedBy(UINT n) {
  if constexpr (std::is_same_v<UINT, uint128_t>) {
    return n / static_cast<UINT>(DENOM);
  } else {
    // G++ can recognize that the reciprocal is a compile-time
    // constant when For() is called inline, but clang requires
    // a constexpr variable definition to force compile-time
    // evaluation of the reciprocal.
    constexpr auto recip{FixedPointReciprocal<UINT>::For(DENOM)};
    return recip.Divide(n);
  }
}
}
#endif
