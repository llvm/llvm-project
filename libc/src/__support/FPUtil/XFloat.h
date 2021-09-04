//===-- Utility class to manipulate wide floats. ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FPBits.h"
#include "NormalFloat.h"
#include "UInt.h"

#include <stdint.h>

namespace __llvm_libc {
namespace fputil {

// Store and manipulate positive double precision numbers at |Precision| bits.
template <size_t Precision> class XFloat {
  static constexpr uint64_t OneMask = (uint64_t(1) << 63);
  UInt<Precision> man;
  static constexpr uint64_t WordCount = Precision / 64;
  int exp;

  size_t bit_width(uint64_t x) {
    if (x == 0)
      return 0;
    size_t shift = 0;
    while ((OneMask & x) == 0) {
      ++shift;
      x <<= 1;
    }
    return 64 - shift;
  }

public:
  XFloat() : exp(0) {
    for (int i = 0; i < WordCount; ++i)
      man[i] = 0;
  }

  XFloat(const XFloat &other) : exp(other.exp) {
    for (int i = 0; i < WordCount; ++i)
      man[i] = other.man[i];
  }

  explicit XFloat(double x) {
    auto xn = NormalFloat<double>(x);
    exp = xn.exponent;
    man[WordCount - 1] = xn.mantissa << 11;
    for (int i = 0; i < WordCount - 1; ++i)
      man[i] = 0;
  }

  XFloat(int e, const UInt<Precision> &bits) : exp(e) {
    for (size_t i = 0; i < WordCount; ++i)
      man[i] = bits[i];
  }

  // Multiply this number with x and store the result in this number.
  void mul(double x) {
    auto xn = NormalFloat<double>(x);
    exp += xn.exponent;
    uint64_t carry = man.mul(xn.mantissa << 11);
    size_t carry_width = bit_width(carry);
    carry_width = (carry_width == 64 ? 64 : 63);
    man.shift_right(carry_width);
    man[WordCount - 1] = man[WordCount - 1] + (carry << (64 - carry_width));
    exp += carry_width == 64 ? 1 : 0;
    normalize();
  }

  void drop_int() {
    if (exp < 0)
      return;
    if (exp > int(Precision - 1)) {
      for (size_t i = 0; i < WordCount; ++i)
        man[i] = 0;
      return;
    }

    man.shift_left(exp + 1);
    man.shift_right(exp + 1);

    normalize();
  }

  double mul(const XFloat<Precision> &other) {
    constexpr size_t row_words = 2 * WordCount + 1;
    constexpr size_t row_precision = row_words * 64;
    constexpr size_t result_bits = 2 * Precision;
    UInt<row_precision> rows[WordCount];

    for (size_t r = 0; r < WordCount; ++r) {
      for (size_t i = 0; i < row_words; ++i) {
        if (i < WordCount)
          rows[r][i] = man[i];
        else
          rows[r][i] = 0;
      }
      rows[r].mul(other.man[r]);
      rows[r].shift_left(r * 64);
    }

    for (size_t r = 1; r < WordCount; ++r) {
      rows[0].add(rows[r]);
    }
    int result_exp = exp + other.exp;
    uint64_t carry = rows[0][row_words - 1];
    if (carry) {
      size_t carry_width = bit_width(carry);
      rows[0].shift_right(carry_width);
      rows[0][row_words - 2] =
          rows[0][row_words - 2] + (carry << (64 - carry_width));
      result_exp += carry_width;
    }

    if (rows[0][row_words - 2] & OneMask) {
      ++result_exp;
    } else {
      rows[0].shift_left(1);
    }

    UInt<result_bits> result_man;
    for (size_t i = 0; i < result_bits / 64; ++i)
      result_man[i] = rows[0][i];
    XFloat<result_bits> result(result_exp, result_man);
    result.normalize();
    return double(result);
  }

  explicit operator double() {
    normalize();

    constexpr uint64_t one = uint64_t(1) << 10;
    constexpr uint64_t excess_mask = (one << 1) - 1;
    uint64_t excess = man[WordCount - 1] & excess_mask;
    uint64_t new_man = man[WordCount - 1] >> 11;
    if (excess > one) {
      // We have to round up.
      ++new_man;
    } else if (excess == one) {
      bool greater_than_one = false;
      for (size_t i = 0; i < WordCount - 1; ++i) {
        greater_than_one = (man[i] != 0);
        if (greater_than_one)
          break;
      }
      if (greater_than_one || (new_man & 1) != 0) {
        ++new_man;
      }
    }

    if (new_man == (uint64_t(1) << 53))
      ++exp;

    // We use NormalFloat as it can produce subnormal numbers or underflow to 0
    // if necessary.
    NormalFloat<double> d(exp, new_man, 0);
    return double(d);
  }

  // Normalizes this number.
  void normalize() {
    uint64_t man_bits = 0;
    for (size_t i = 0; i < WordCount; ++i)
      man_bits |= man[i];

    if (man_bits == 0)
      return;

    while ((man[WordCount - 1] & OneMask) == 0) {
      man.shift_left(1);
      --exp;
    }
  }
};

} // namespace fputil
} // namespace __llvm_libc
