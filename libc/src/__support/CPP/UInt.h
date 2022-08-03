//===-- A class to manipulate wide integers. --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_UTILS_CPP_UINT_H
#define LLVM_LIBC_UTILS_CPP_UINT_H

#include "array.h"

#include <stddef.h> // For size_t
#include <stdint.h>

namespace __llvm_libc {
namespace cpp {

template <size_t Bits> class UInt {

  static_assert(Bits > 0 && Bits % 64 == 0,
                "Number of bits in UInt should be a multiple of 64.");
  static constexpr size_t WordCount = Bits / 64;
  uint64_t val[WordCount];

  static constexpr uint64_t MASK32 = 0xFFFFFFFFu;

  static constexpr uint64_t low(uint64_t v) { return v & MASK32; }
  static constexpr uint64_t high(uint64_t v) { return (v >> 32) & MASK32; }

public:
  constexpr UInt() {}

  constexpr UInt(const UInt<Bits> &other) {
    for (size_t i = 0; i < WordCount; ++i)
      val[i] = other.val[i];
  }

  // Initialize the first word to |v| and the rest to 0.
  constexpr UInt(uint64_t v) {
    val[0] = v;
    for (size_t i = 1; i < WordCount; ++i) {
      val[i] = 0;
    }
  }
  constexpr explicit UInt(const cpp::array<uint64_t, WordCount> &words) {
    for (size_t i = 0; i < WordCount; ++i)
      val[i] = words[i];
  }

  constexpr explicit operator uint64_t() const { return val[0]; }

  constexpr explicit operator uint32_t() const {
    return uint32_t(uint64_t(*this));
  }

  constexpr explicit operator uint8_t() const {
    return uint8_t(uint64_t(*this));
  }

  UInt<Bits> &operator=(const UInt<Bits> &other) {
    for (size_t i = 0; i < WordCount; ++i)
      val[i] = other.val[i];
    return *this;
  }

  // Add x to this number and store the result in this number.
  // Returns the carry value produced by the addition operation.
  // To prevent overflow from intermediate results, we use the following
  // property of unsigned integers:
  //   x + (~x) = 2^(sizeof(x)) - 1.
  constexpr uint64_t add(const UInt<Bits> &x) {
    bool carry = false;
    for (size_t i = 0; i < WordCount; ++i) {
      uint64_t complement = ~x.val[i];
      if (!carry) {
        if (val[i] <= complement)
          val[i] += x.val[i];
        else {
          val[i] -= complement + 1;
          carry = true;
        }
      } else {
        if (val[i] < complement) {
          val[i] += x.val[i] + 1;
          carry = false;
        } else
          val[i] -= complement;
      }
    }
    return carry ? 1 : 0;
  }

  constexpr UInt<Bits> operator+(const UInt<Bits> &other) const {
    UInt<Bits> result(*this);
    result.add(other);
    // TODO(lntue): Set overflow flag / errno when carry is true.
    return result;
  }

  constexpr UInt<Bits> operator+=(const UInt<Bits> &other) {
    // TODO(lntue): Set overflow flag / errno when carry is true.
    add(other);
    return *this;
  }

  // Multiply this number with x and store the result in this number. It is
  // implemented using the long multiplication algorithm by splitting the
  // 64-bit words of this number and |x| in to 32-bit halves but peforming
  // the operations using 64-bit numbers. This ensures that we don't lose the
  // carry bits.
  // Returns the carry value produced by the multiplication operation.
  constexpr uint64_t mul(uint64_t x) {
    uint64_t x_lo = low(x);
    uint64_t x_hi = high(x);

    cpp::array<uint64_t, WordCount + 1> row1;
    uint64_t carry = 0;
    for (size_t i = 0; i < WordCount; ++i) {
      uint64_t l = low(val[i]);
      uint64_t h = high(val[i]);
      uint64_t p1 = x_lo * l;
      uint64_t p2 = x_lo * h;

      uint64_t res_lo = low(p1) + carry;
      carry = high(res_lo);
      uint64_t res_hi = high(p1) + low(p2) + carry;
      carry = high(res_hi) + high(p2);

      res_lo = low(res_lo);
      res_hi = low(res_hi);
      row1[i] = res_lo + (res_hi << 32);
    }
    row1[WordCount] = carry;

    cpp::array<uint64_t, WordCount + 1> row2;
    row2[0] = 0;
    carry = 0;
    for (size_t i = 0; i < WordCount; ++i) {
      uint64_t l = low(val[i]);
      uint64_t h = high(val[i]);
      uint64_t p1 = x_hi * l;
      uint64_t p2 = x_hi * h;

      uint64_t res_lo = low(p1) + carry;
      carry = high(res_lo);
      uint64_t res_hi = high(p1) + low(p2) + carry;
      carry = high(res_hi) + high(p2);

      res_lo = low(res_lo);
      res_hi = low(res_hi);
      row2[i] = res_lo + (res_hi << 32);
    }
    row2[WordCount] = carry;

    UInt<(WordCount + 1) * 64> r1(row1), r2(row2);
    r2.shift_left(32);
    r1.add(r2);
    for (size_t i = 0; i < WordCount; ++i) {
      val[i] = r1[i];
    }
    return r1[WordCount];
  }

  constexpr UInt<Bits> operator*(const UInt<Bits> &other) const {
    UInt<Bits> result(0);
    for (size_t i = 0; i < WordCount; ++i) {
      UInt<Bits> row_result(*this);
      row_result.mul(other[i]);
      row_result.shift_left(64 * i);
      result = result + row_result;
    }
    return result;
  }

  constexpr UInt<Bits> &operator*=(const UInt<Bits> &other) {
    *this = *this * other;
    return *this;
  }

  constexpr void shift_left(size_t s) {
    const size_t drop = s / 64;  // Number of words to drop
    const size_t shift = s % 64; // Bits to shift in the remaining words.
    const uint64_t mask = ((uint64_t(1) << shift) - 1) << (64 - shift);

    for (size_t i = WordCount; drop > 0 && i > 0; --i) {
      if (i > drop)
        val[i - 1] = val[i - drop - 1];
      else
        val[i - 1] = 0;
    }
    for (size_t i = WordCount; shift > 0 && i > drop; --i) {
      uint64_t drop_val = (val[i - 1] & mask) >> (64 - shift);
      val[i - 1] <<= shift;
      if (i < WordCount)
        val[i] |= drop_val;
    }
  }

  constexpr UInt<Bits> operator<<(size_t s) const {
    UInt<Bits> result(*this);
    result.shift_left(s);
    return result;
  }

  constexpr UInt<Bits> &operator<<=(size_t s) {
    shift_left(s);
    return *this;
  }

  constexpr void shift_right(size_t s) {
    const size_t drop = s / 64;  // Number of words to drop
    const size_t shift = s % 64; // Bit shift in the remaining words.
    const uint64_t mask = (uint64_t(1) << shift) - 1;

    for (size_t i = 0; drop > 0 && i < WordCount; ++i) {
      if (i + drop < WordCount)
        val[i] = val[i + drop];
      else
        val[i] = 0;
    }
    for (size_t i = 0; shift > 0 && i < WordCount; ++i) {
      uint64_t drop_val = ((val[i] & mask) << (64 - shift));
      val[i] >>= shift;
      if (i > 0)
        val[i - 1] |= drop_val;
    }
  }

  constexpr UInt<Bits> operator>>(size_t s) const {
    UInt<Bits> result(*this);
    result.shift_right(s);
    return result;
  }

  constexpr UInt<Bits> &operator>>=(size_t s) {
    shift_right(s);
    return *this;
  }

  constexpr UInt<Bits> operator&(const UInt<Bits> &other) const {
    UInt<Bits> result;
    for (size_t i = 0; i < WordCount; ++i)
      result.val[i] = val[i] & other.val[i];
    return result;
  }

  constexpr UInt<Bits> &operator&=(const UInt<Bits> &other) {
    for (size_t i = 0; i < WordCount; ++i)
      val[i] &= other.val[i];
    return *this;
  }

  constexpr UInt<Bits> operator|(const UInt<Bits> &other) const {
    UInt<Bits> result;
    for (size_t i = 0; i < WordCount; ++i)
      result.val[i] = val[i] | other.val[i];
    return result;
  }

  constexpr UInt<Bits> &operator|=(const UInt<Bits> &other) {
    for (size_t i = 0; i < WordCount; ++i)
      val[i] |= other.val[i];
    return *this;
  }

  constexpr UInt<Bits> operator^(const UInt<Bits> &other) const {
    UInt<Bits> result;
    for (size_t i = 0; i < WordCount; ++i)
      result.val[i] = val[i] ^ other.val[i];
    return result;
  }

  constexpr UInt<Bits> &operator^=(const UInt<Bits> &other) {
    for (size_t i = 0; i < WordCount; ++i)
      val[i] ^= other.val[i];
    return *this;
  }

  constexpr UInt<Bits> operator~() const {
    UInt<Bits> result;
    for (size_t i = 0; i < WordCount; ++i)
      result.val[i] = ~val[i];
    return result;
  }

  constexpr bool operator==(const UInt<Bits> &other) const {
    for (size_t i = 0; i < WordCount; ++i) {
      if (val[i] != other.val[i])
        return false;
    }
    return true;
  }

  constexpr bool operator!=(const UInt<Bits> &other) const {
    for (size_t i = 0; i < WordCount; ++i) {
      if (val[i] != other.val[i])
        return true;
    }
    return false;
  }

  constexpr bool operator>(const UInt<Bits> &other) const {
    for (size_t i = WordCount; i > 0; --i) {
      uint64_t word = val[i - 1];
      uint64_t other_word = other.val[i - 1];
      if (word > other_word)
        return true;
      else if (word < other_word)
        return false;
    }
    // Equal
    return false;
  }

  constexpr bool operator>=(const UInt<Bits> &other) const {
    for (size_t i = WordCount; i > 0; --i) {
      uint64_t word = val[i - 1];
      uint64_t other_word = other.val[i - 1];
      if (word > other_word)
        return true;
      else if (word < other_word)
        return false;
    }
    // Equal
    return true;
  }

  constexpr bool operator<(const UInt<Bits> &other) const {
    for (size_t i = WordCount; i > 0; --i) {
      uint64_t word = val[i - 1];
      uint64_t other_word = other.val[i - 1];
      if (word > other_word)
        return false;
      else if (word < other_word)
        return true;
    }
    // Equal
    return false;
  }

  constexpr bool operator<=(const UInt<Bits> &other) const {
    for (size_t i = WordCount; i > 0; --i) {
      uint64_t word = val[i - 1];
      uint64_t other_word = other.val[i - 1];
      if (word > other_word)
        return false;
      else if (word < other_word)
        return true;
    }
    // Equal
    return true;
  }

  constexpr UInt<Bits> &operator++() {
    UInt<Bits> one(1);
    add(one);
    return *this;
  }

  // Return the i-th 64-bit word of the number.
  constexpr const uint64_t &operator[](size_t i) const { return val[i]; }

  // Return the i-th 64-bit word of the number.
  constexpr uint64_t &operator[](size_t i) { return val[i]; }

  uint64_t *data() { return val; }

  const uint64_t *data() const { return val; }
};

template <>
constexpr UInt<128> UInt<128>::operator*(const UInt<128> &other) const {
  // temp low covers bits 0-63, middle covers 32-95, high covers 64-127, and
  // high overflow covers 96-159.
  uint64_t temp_low = low(val[0]) * low(other[0]);
  uint64_t temp_middle_1 = low(val[0]) * high(other[0]);
  uint64_t temp_middle_2 = high(val[0]) * low(other[0]);

  // temp_middle is split out so that overflows can be handled, but since
  // but since the result will be truncated to 128 bits any overflow from here
  // on doesn't matter.
  uint64_t temp_high = low(val[0]) * low(other[1]) +
                       high(val[0]) * high(other[0]) +
                       low(val[1]) * low(other[0]);

  uint64_t temp_high_overflow =
      low(val[0]) * high(other[1]) + high(val[0]) * low(other[1]) +
      low(val[1]) * high(other[0]) + high(val[1]) * low(other[0]);

  // temp_low_middle has just the high 32 bits of low, as well as any
  // overflow.
  uint64_t temp_low_middle =
      high(temp_low) + low(temp_middle_1) + low(temp_middle_2);

  uint64_t new_low = low(temp_low) + (low(temp_low_middle) << 32);
  uint64_t new_high = high(temp_low_middle) + high(temp_middle_1) +
                      high(temp_middle_2) + temp_high +
                      (low(temp_high_overflow) << 32);
  UInt<128> result(0);
  result[0] = new_low;
  result[1] = new_high;
  return result;
}

} // namespace cpp
} // namespace __llvm_libc

#endif // LLVM_LIBC_UTILS_CPP_UINT_H
