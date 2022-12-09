//===-- A class to manipulate wide integers. --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_UINT_H
#define LLVM_LIBC_SRC_SUPPORT_UINT_H

#include "src/__support/CPP/array.h"
#include "src/__support/CPP/limits.h"
#include "src/__support/CPP/optional.h"
#include "src/__support/CPP/type_traits.h"
#include "src/__support/builtin_wrappers.h"
#include "src/__support/integer_utils.h"
#include "src/__support/number_pair.h"

#include <stddef.h> // For size_t
#include <stdint.h>

namespace __llvm_libc::cpp {

template <size_t Bits> struct UInt {

  static_assert(Bits > 0 && Bits % 64 == 0,
                "Number of bits in UInt should be a multiple of 64.");
  static constexpr size_t WordCount = Bits / 64;
  uint64_t val[WordCount];

  static constexpr uint64_t MASK32 = 0xFFFFFFFFu;

  static constexpr uint64_t low(uint64_t v) { return v & MASK32; }
  static constexpr uint64_t high(uint64_t v) { return (v >> 32) & MASK32; }

  constexpr UInt() {}

  constexpr UInt(const UInt<Bits> &other) {
    for (size_t i = 0; i < WordCount; ++i)
      val[i] = other.val[i];
  }

  template <size_t OtherBits> constexpr UInt(const UInt<OtherBits> &other) {
    if (OtherBits >= Bits) {
      for (size_t i = 0; i < WordCount; ++i)
        val[i] = other[i];
    } else {
      size_t i = 0;
      for (; i < OtherBits / 64; ++i)
        val[i] = other[i];
      for (; i < WordCount; ++i)
        val[i] = 0;
    }
  }

  // Construct a UInt from a C array.
  template <size_t N, enable_if_t<N <= WordCount, int> = 0>
  constexpr UInt(const uint64_t (&nums)[N]) {
    size_t min_wordcount = N < WordCount ? N : WordCount;
    size_t i = 0;
    for (; i < min_wordcount; ++i)
      val[i] = nums[i];

    // If nums doesn't completely fill val, then fill the rest with zeroes.
    for (; i < WordCount; ++i)
      val[i] = 0;
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
    SumCarry<uint64_t> s{0, 0};
    for (size_t i = 0; i < WordCount; ++i) {
      s = add_with_carry(val[i], x.val[i], s.carry);
      val[i] = s.sum;
    }
    return s.carry;
  }

  constexpr UInt<Bits> operator+(const UInt<Bits> &other) const {
    UInt<Bits> result;
    SumCarry<uint64_t> s{0, 0};
    for (size_t i = 0; i < WordCount; ++i) {
      s = add_with_carry(val[i], other.val[i], s.carry);
      result.val[i] = s.sum;
    }
    return result;
  }

  constexpr UInt<Bits> operator+=(const UInt<Bits> &other) {
    add(other); // Returned carry value is ignored.
    return *this;
  }

  // Subtract x to this number and store the result in this number.
  // Returns the carry value produced by the subtraction operation.
  // To prevent overflow from intermediate results, we use the following
  // property of unsigned integers:
  //   x + (~x) = 2^(sizeof(x)) - 1,
  // So:
  //   -x = ((~x) + 1) + (-2^(sizeof(x))),
  // where 2^(sizeof(x)) is represented by the carry bit.
  constexpr uint64_t sub(const UInt<Bits> &x) {
    bool carry = false;
    for (size_t i = 0; i < WordCount; ++i) {
      if (!carry) {
        if (val[i] >= x.val[i])
          val[i] -= x.val[i];
        else {
          val[i] += (~x.val[i]) + 1;
          carry = true;
        }
      } else {
        if (val[i] > x.val[i]) {
          val[i] -= x.val[i] + 1;
          carry = false;
        } else
          val[i] += ~x.val[i];
      }
    }
    return carry ? 1 : 0;
  }

  constexpr UInt<Bits> operator-(const UInt<Bits> &other) const {
    UInt<Bits> result(*this);
    result.sub(other);
    // TODO(lntue): Set overflow flag / errno when carry is true.
    return result;
  }

  constexpr UInt<Bits> operator-=(const UInt<Bits> &other) {
    // TODO(lntue): Set overflow flag / errno when carry is true.
    sub(other);
    return *this;
  }

  // Multiply this number with x and store the result in this number. It is
  // implemented using the long multiplication algorithm by splitting the
  // 64-bit words of this number and |x| in to 32-bit halves but peforming
  // the operations using 64-bit numbers. This ensures that we don't lose the
  // carry bits.
  // Returns the carry value produced by the multiplication operation.
  constexpr uint64_t mul(uint64_t x) {
    UInt<128> partial_sum(0);
    uint64_t carry = 0;
    for (size_t i = 0; i < WordCount; ++i) {
      NumberPair<uint64_t> prod = full_mul(val[i], x);
      UInt<128> tmp({prod.lo, prod.hi});
      carry += partial_sum.add(tmp);
      val[i] = partial_sum.val[0];
      partial_sum.val[0] = partial_sum.val[1];
      partial_sum.val[1] = carry;
      carry = 0;
    }
    return partial_sum.val[1];
  }

  constexpr UInt<Bits> operator*(const UInt<Bits> &other) const {
    if constexpr (WordCount == 1) {
      return {val[0] * other.val[0]};
    } else {
      UInt<Bits> result(0);
      UInt<128> partial_sum(0);
      uint64_t carry = 0;
      for (size_t i = 0; i < WordCount; ++i) {
        for (size_t j = 0; j <= i; j++) {
          NumberPair<uint64_t> prod = full_mul(val[j], other.val[i - j]);
          UInt<128> tmp({prod.lo, prod.hi});
          carry += partial_sum.add(tmp);
        }
        result.val[i] = partial_sum.val[0];
        partial_sum.val[0] = partial_sum.val[1];
        partial_sum.val[1] = carry;
        carry = 0;
      }
      return result;
    }
  }

  // pow takes a power and sets this to its starting value to that power. Zero
  // to the zeroth power returns 1.
  constexpr void pow_n(uint64_t power) {
    UInt<Bits> result = 1;
    UInt<Bits> cur_power = *this;

    while (power > 0) {
      if ((power % 2) > 0) {
        result = result * cur_power;
      }
      power = power >> 1;
      cur_power *= cur_power;
    }
    *this = result;
  }

  // div takes another UInt of the same size and divides this by it. The value
  // of this will be set to the quotient, and the return value is the remainder.
  constexpr optional<UInt<Bits>> div(const UInt<Bits> &other) {
    UInt<Bits> remainder(0);
    if (*this < other) {
      remainder = *this;
      *this = UInt<Bits>(0);
      return remainder;
    }
    if (other == 1) {
      return remainder;
    }
    if (other == 0) {
      return nullopt;
    }

    UInt<Bits> quotient(0);
    UInt<Bits> subtractor = other;
    int cur_bit = subtractor.clz() - this->clz();
    subtractor.shift_left(cur_bit);

    for (; cur_bit >= 0 && *this > 0; --cur_bit, subtractor.shift_right(1)) {
      if (*this >= subtractor) {
        this->sub(subtractor);
        quotient = quotient | (UInt<Bits>(1) << cur_bit);
      }
    }
    remainder = *this;
    *this = quotient;
    return remainder;
  }

  constexpr UInt<Bits> operator/(const UInt<Bits> &other) const {
    UInt<Bits> result(*this);
    result.div(other);
    return result;
  }

  constexpr UInt<Bits> operator%(const UInt<Bits> &other) const {
    UInt<Bits> result(*this);
    return *result.div(other);
  }

  constexpr UInt<Bits> &operator*=(const UInt<Bits> &other) {
    *this = *this * other;
    return *this;
  }

  constexpr uint64_t clz() {
    uint64_t leading_zeroes = 0;
    for (size_t i = WordCount; i > 0; --i) {
      if (val[i - 1] == 0) {
        leading_zeroes += sizeof(uint64_t) * 8;
      } else {
        leading_zeroes += unsafe_clz(val[i - 1]);
        break;
      }
    }
    return leading_zeroes;
  }

  constexpr void shift_left(size_t s) {
#ifdef __SIZEOF_INT128__
    if constexpr (Bits == 128) {
      // Use builtin 128 bits if available;
      if (s >= 128) {
        val[0] = 0;
        val[1] = 0;
        return;
      }
      __uint128_t tmp = __uint128_t(val[0]) + (__uint128_t(val[1]) << 64);
      tmp <<= s;
      val[0] = uint64_t(tmp);
      val[1] = uint64_t(tmp >> 64);
      return;
    }
#endif                           // __SIZEOF_INT128__
    const size_t drop = s / 64;  // Number of words to drop
    const size_t shift = s % 64; // Bits to shift in the remaining words.
    size_t i = WordCount;

    if (drop < WordCount) {
      i = WordCount - 1;
      size_t j = WordCount - 1 - drop;
      for (; j > 0; --i, --j) {
        val[i] = (val[j] << shift) | (val[j - 1] >> (64 - shift));
      }
      val[i] = val[0] << shift;
    }

    for (size_t j = 0; j < i; ++j) {
      val[j] = 0;
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

    size_t i = 0;

    if (drop < WordCount) {
      size_t j = drop;
      for (; j < WordCount - 1; ++i, ++j) {
        val[i] = (val[j] >> shift) | (val[j + 1] << (64 - shift));
      }
      val[i] = val[j] >> shift;
      ++i;
    }

    for (; i < WordCount; ++i) {
      val[i] = 0;
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

// Provides limits of UInt<128>.
template <> class numeric_limits<UInt<128>> {
public:
  static constexpr UInt<128> max() { return ~UInt<128>(0); }
  static constexpr UInt<128> min() { return 0; }
};

// Provides is_integral of UInt<128>, UInt<192>, UInt<256>.
template <size_t Bits> struct is_integral<UInt<Bits>> : public cpp::true_type {
  static_assert(Bits > 0 && Bits % 64 == 0,
                "Number of bits in UInt should be a multiple of 64.");
};

// Provides is_unsigned of UInt<128>, UInt<192>, UInt<256>.
template <size_t Bits> struct is_unsigned<UInt<Bits>> : public cpp::true_type {
  static_assert(Bits > 0 && Bits % 64 == 0,
                "Number of bits in UInt should be a multiple of 64.");
};

} // namespace __llvm_libc::cpp

#endif // LLVM_LIBC_SRC_SUPPORT_UINT_H
