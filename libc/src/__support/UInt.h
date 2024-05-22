//===-- A class to manipulate wide integers. --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_UINT_H
#define LLVM_LIBC_SRC___SUPPORT_UINT_H

#include "src/__support/CPP/array.h"
#include "src/__support/CPP/bit.h" // countl_zero
#include "src/__support/CPP/limits.h"
#include "src/__support/CPP/optional.h"
#include "src/__support/CPP/type_traits.h"
#include "src/__support/integer_utils.h"
#include "src/__support/macros/attributes.h"   // LIBC_INLINE
#include "src/__support/macros/optimization.h" // LIBC_UNLIKELY
#include "src/__support/math_extras.h"         // SumCarry, DiffBorrow
#include "src/__support/number_pair.h"

#include <stddef.h> // For size_t
#include <stdint.h>

namespace LIBC_NAMESPACE::cpp {

template <size_t Bits, bool Signed> struct BigInt {

  static_assert(Bits > 0 && Bits % 64 == 0,
                "Number of bits in BigInt should be a multiple of 64.");
  LIBC_INLINE_VAR static constexpr size_t WORDCOUNT = Bits / 64;
  uint64_t val[WORDCOUNT]{};

  LIBC_INLINE_VAR static constexpr uint64_t MASK32 = 0xFFFFFFFFu;

  LIBC_INLINE static constexpr uint64_t low(uint64_t v) { return v & MASK32; }
  LIBC_INLINE static constexpr uint64_t high(uint64_t v) {
    return (v >> 32) & MASK32;
  }

  LIBC_INLINE constexpr BigInt() = default;

  LIBC_INLINE constexpr BigInt(const BigInt<Bits, Signed> &other) = default;

  template <size_t OtherBits, bool OtherSigned>
  LIBC_INLINE constexpr BigInt(const BigInt<OtherBits, OtherSigned> &other) {
    if (OtherBits >= Bits) {
      for (size_t i = 0; i < WORDCOUNT; ++i)
        val[i] = other[i];
    } else {
      size_t i = 0;
      for (; i < OtherBits / 64; ++i)
        val[i] = other[i];
      uint64_t sign = 0;
      if constexpr (Signed && OtherSigned) {
        sign = static_cast<uint64_t>(
            -static_cast<int64_t>(other[OtherBits / 64 - 1] >> 63));
      }
      for (; i < WORDCOUNT; ++i)
        val[i] = sign;
    }
  }

  // Construct a BigInt from a C array.
  template <size_t N, enable_if_t<N <= WORDCOUNT, int> = 0>
  LIBC_INLINE constexpr BigInt(const uint64_t (&nums)[N]) {
    size_t min_wordcount = N < WORDCOUNT ? N : WORDCOUNT;
    size_t i = 0;
    for (; i < min_wordcount; ++i)
      val[i] = nums[i];

    // If nums doesn't completely fill val, then fill the rest with zeroes.
    for (; i < WORDCOUNT; ++i)
      val[i] = 0;
  }

  // Initialize the first word to |v| and the rest to 0.
  template <typename T,
            typename = cpp::enable_if_t<is_integral_v<T> && sizeof(T) <= 16>>
  LIBC_INLINE constexpr BigInt(T v) {
    val[0] = static_cast<uint64_t>(v);

    if constexpr (Bits == 64)
      return;

    // Bits is at least 128.
    size_t i = 1;
    if constexpr (sizeof(T) == 16) {
      val[1] = static_cast<uint64_t>(v >> 64);
      i = 2;
    }

    uint64_t sign = (Signed && (v < 0)) ? 0xffff'ffff'ffff'ffff : 0;
    for (; i < WORDCOUNT; ++i) {
      val[i] = sign;
    }
  }

  LIBC_INLINE constexpr explicit BigInt(
      const cpp::array<uint64_t, WORDCOUNT> &words) {
    for (size_t i = 0; i < WORDCOUNT; ++i)
      val[i] = words[i];
  }

  template <typename T> LIBC_INLINE constexpr explicit operator T() const {
    return to<T>();
  }

  template <typename T>
  LIBC_INLINE constexpr cpp::enable_if_t<
      cpp::is_integral_v<T> && sizeof(T) <= 8 && !cpp::is_same_v<T, bool>, T>
  to() const {
    return static_cast<T>(val[0]);
  }
  template <typename T>
  LIBC_INLINE constexpr cpp::enable_if_t<
      cpp::is_integral_v<T> && sizeof(T) == 16, T>
  to() const {
    // T is 128-bit.
    T lo = static_cast<T>(val[0]);

    if constexpr (Bits == 64) {
      if constexpr (Signed) {
        // Extend sign for negative numbers.
        return (val[0] >> 63) ? ((T(-1) << 64) + lo) : lo;
      } else {
        return lo;
      }
    } else {
      return static_cast<T>((static_cast<T>(val[1]) << 64) + lo);
    }
  }

  LIBC_INLINE constexpr explicit operator bool() const { return !is_zero(); }

  LIBC_INLINE BigInt<Bits, Signed> &
  operator=(const BigInt<Bits, Signed> &other) = default;

  LIBC_INLINE constexpr bool is_zero() const {
    for (size_t i = 0; i < WORDCOUNT; ++i) {
      if (val[i] != 0)
        return false;
    }
    return true;
  }

  // Add x to this number and store the result in this number.
  // Returns the carry value produced by the addition operation.
  LIBC_INLINE constexpr uint64_t add(const BigInt<Bits, Signed> &x) {
    SumCarry<uint64_t> s{0, 0};
    for (size_t i = 0; i < WORDCOUNT; ++i) {
      s = add_with_carry_const(val[i], x.val[i], s.carry);
      val[i] = s.sum;
    }
    return s.carry;
  }

  LIBC_INLINE constexpr BigInt<Bits, Signed>
  operator+(const BigInt<Bits, Signed> &other) const {
    BigInt<Bits, Signed> result;
    SumCarry<uint64_t> s{0, 0};
    for (size_t i = 0; i < WORDCOUNT; ++i) {
      s = add_with_carry(val[i], other.val[i], s.carry);
      result.val[i] = s.sum;
    }
    return result;
  }

  // This will only apply when initializing a variable from constant values, so
  // it will always use the constexpr version of add_with_carry.
  LIBC_INLINE constexpr BigInt<Bits, Signed>
  operator+(BigInt<Bits, Signed> &&other) const {
    BigInt<Bits, Signed> result;
    SumCarry<uint64_t> s{0, 0};
    for (size_t i = 0; i < WORDCOUNT; ++i) {
      s = add_with_carry_const(val[i], other.val[i], s.carry);
      result.val[i] = s.sum;
    }
    return result;
  }

  LIBC_INLINE constexpr BigInt<Bits, Signed> &
  operator+=(const BigInt<Bits, Signed> &other) {
    add(other); // Returned carry value is ignored.
    return *this;
  }

  // Subtract x to this number and store the result in this number.
  // Returns the carry value produced by the subtraction operation.
  LIBC_INLINE constexpr uint64_t sub(const BigInt<Bits, Signed> &x) {
    DiffBorrow<uint64_t> d{0, 0};
    for (size_t i = 0; i < WORDCOUNT; ++i) {
      d = sub_with_borrow_const(val[i], x.val[i], d.borrow);
      val[i] = d.diff;
    }
    return d.borrow;
  }

  LIBC_INLINE BigInt<Bits, Signed>
  operator-(const BigInt<Bits, Signed> &other) const {
    BigInt<Bits, Signed> result;
    DiffBorrow<uint64_t> d{0, 0};
    for (size_t i = 0; i < WORDCOUNT; ++i) {
      d = sub_with_borrow(val[i], other.val[i], d.borrow);
      result.val[i] = d.diff;
    }
    return result;
  }

  LIBC_INLINE constexpr BigInt<Bits, Signed>
  operator-(BigInt<Bits, Signed> &&other) const {
    BigInt<Bits, Signed> result;
    DiffBorrow<uint64_t> d{0, 0};
    for (size_t i = 0; i < WORDCOUNT; ++i) {
      d = sub_with_borrow_const(val[i], other.val[i], d.borrow);
      result.val[i] = d.diff;
    }
    return result;
  }

  LIBC_INLINE constexpr BigInt<Bits, Signed> &
  operator-=(const BigInt<Bits, Signed> &other) {
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
  LIBC_INLINE constexpr uint64_t mul(uint64_t x) {
    BigInt<128, Signed> partial_sum(0);
    uint64_t carry = 0;
    for (size_t i = 0; i < WORDCOUNT; ++i) {
      NumberPair<uint64_t> prod = full_mul(val[i], x);
      BigInt<128, Signed> tmp({prod.lo, prod.hi});
      carry += partial_sum.add(tmp);
      val[i] = partial_sum.val[0];
      partial_sum.val[0] = partial_sum.val[1];
      partial_sum.val[1] = carry;
      carry = 0;
    }
    return partial_sum.val[1];
  }

  LIBC_INLINE constexpr BigInt<Bits, Signed>
  operator*(const BigInt<Bits, Signed> &other) const {
    if constexpr (Signed) {
      BigInt<Bits, false> a(*this);
      BigInt<Bits, false> b(other);
      bool a_neg = (a.val[WORDCOUNT - 1] >> 63);
      bool b_neg = (b.val[WORDCOUNT - 1] >> 63);
      if (a_neg)
        a = -a;
      if (b_neg)
        b = -b;
      BigInt<Bits, false> prod = a * b;
      if (a_neg != b_neg)
        prod = -prod;
      return static_cast<BigInt<Bits, true>>(prod);
    } else {

      if constexpr (WORDCOUNT == 1) {
        return {val[0] * other.val[0]};
      } else {
        BigInt<Bits, Signed> result(0);
        BigInt<128, Signed> partial_sum(0);
        uint64_t carry = 0;
        for (size_t i = 0; i < WORDCOUNT; ++i) {
          for (size_t j = 0; j <= i; j++) {
            NumberPair<uint64_t> prod = full_mul(val[j], other.val[i - j]);
            BigInt<128, Signed> tmp({prod.lo, prod.hi});
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
  }

  // Return the full product, only unsigned for now.
  template <size_t OtherBits>
  LIBC_INLINE constexpr BigInt<Bits + OtherBits, Signed>
  ful_mul(const BigInt<OtherBits, Signed> &other) const {
    BigInt<Bits + OtherBits, Signed> result(0);
    BigInt<128, Signed> partial_sum(0);
    uint64_t carry = 0;
    constexpr size_t OTHER_WORDCOUNT = BigInt<OtherBits, Signed>::WORDCOUNT;
    for (size_t i = 0; i <= WORDCOUNT + OTHER_WORDCOUNT - 2; ++i) {
      const size_t lower_idx =
          i < OTHER_WORDCOUNT ? 0 : i - OTHER_WORDCOUNT + 1;
      const size_t upper_idx = i < WORDCOUNT ? i : WORDCOUNT - 1;
      for (size_t j = lower_idx; j <= upper_idx; ++j) {
        NumberPair<uint64_t> prod = full_mul(val[j], other.val[i - j]);
        BigInt<128, Signed> tmp({prod.lo, prod.hi});
        carry += partial_sum.add(tmp);
      }
      result.val[i] = partial_sum.val[0];
      partial_sum.val[0] = partial_sum.val[1];
      partial_sum.val[1] = carry;
      carry = 0;
    }
    result.val[WORDCOUNT + OTHER_WORDCOUNT - 1] = partial_sum.val[0];
    return result;
  }

  // Fast hi part of the full product.  The normal product `operator*` returns
  // `Bits` least significant bits of the full product, while this function will
  // approximate `Bits` most significant bits of the full product with errors
  // bounded by:
  //   0 <= (a.full_mul(b) >> Bits) - a.quick_mul_hi(b)) <= WORDCOUNT - 1.
  //
  // An example usage of this is to quickly (but less accurately) compute the
  // product of (normalized) mantissas of floating point numbers:
  //   (mant_1, mant_2) -> quick_mul_hi -> normalize leading bit
  // is much more efficient than:
  //   (mant_1, mant_2) -> ful_mul -> normalize leading bit
  //                    -> convert back to same Bits width by shifting/rounding,
  // especially for higher precisions.
  //
  // Performance summary:
  //   Number of 64-bit x 64-bit -> 128-bit multiplications performed.
  //   Bits  WORDCOUNT  ful_mul  quick_mul_hi  Error bound
  //    128      2         4           3            1
  //    196      3         9           6            2
  //    256      4        16          10            3
  //    512      8        64          36            7
  LIBC_INLINE constexpr BigInt<Bits, Signed>
  quick_mul_hi(const BigInt<Bits, Signed> &other) const {
    BigInt<Bits, Signed> result(0);
    BigInt<128, Signed> partial_sum(0);
    uint64_t carry = 0;
    // First round of accumulation for those at WORDCOUNT - 1 in the full
    // product.
    for (size_t i = 0; i < WORDCOUNT; ++i) {
      NumberPair<uint64_t> prod =
          full_mul(val[i], other.val[WORDCOUNT - 1 - i]);
      BigInt<128, Signed> tmp({prod.lo, prod.hi});
      carry += partial_sum.add(tmp);
    }
    for (size_t i = WORDCOUNT; i < 2 * WORDCOUNT - 1; ++i) {
      partial_sum.val[0] = partial_sum.val[1];
      partial_sum.val[1] = carry;
      carry = 0;
      for (size_t j = i - WORDCOUNT + 1; j < WORDCOUNT; ++j) {
        NumberPair<uint64_t> prod = full_mul(val[j], other.val[i - j]);
        BigInt<128, Signed> tmp({prod.lo, prod.hi});
        carry += partial_sum.add(tmp);
      }
      result.val[i - WORDCOUNT] = partial_sum.val[0];
    }
    result.val[WORDCOUNT - 1] = partial_sum.val[1];
    return result;
  }

  // pow takes a power and sets this to its starting value to that power. Zero
  // to the zeroth power returns 1.
  LIBC_INLINE constexpr void pow_n(uint64_t power) {
    BigInt<Bits, Signed> result = 1;
    BigInt<Bits, Signed> cur_power = *this;

    while (power > 0) {
      if ((power % 2) > 0) {
        result = result * cur_power;
      }
      power = power >> 1;
      cur_power *= cur_power;
    }
    *this = result;
  }

  // TODO: Make division work correctly for signed integers.

  // div takes another BigInt of the same size and divides this by it. The value
  // of this will be set to the quotient, and the return value is the remainder.
  LIBC_INLINE constexpr optional<BigInt<Bits, Signed>>
  div(const BigInt<Bits, Signed> &other) {
    BigInt<Bits, Signed> remainder(0);
    if (*this < other) {
      remainder = *this;
      *this = BigInt<Bits, Signed>(0);
      return remainder;
    }
    if (other == 1) {
      return remainder;
    }
    if (other == 0) {
      return nullopt;
    }

    BigInt<Bits, Signed> quotient(0);
    BigInt<Bits, Signed> subtractor = other;
    int cur_bit = static_cast<int>(subtractor.clz() - this->clz());
    subtractor.shift_left(cur_bit);

    for (; cur_bit >= 0 && *this > 0; --cur_bit, subtractor.shift_right(1)) {
      if (*this >= subtractor) {
        this->sub(subtractor);
        quotient = quotient | (BigInt<Bits, Signed>(1) << cur_bit);
      }
    }
    remainder = *this;
    *this = quotient;
    return remainder;
  }

  // Efficiently perform BigInt / (x * 2^e), where x is a 32-bit unsigned
  // integer, and return the remainder. The main idea is as follow:
  //   Let q = y / (x * 2^e) be the quotient, and
  //       r = y % (x * 2^e) be the remainder.
  //   First, notice that:
  //     r % (2^e) = y % (2^e),
  // so we just need to focus on all the bits of y that is >= 2^e.
  //   To speed up the shift-and-add steps, we only use x as the divisor, and
  // performing 32-bit shiftings instead of bit-by-bit shiftings.
  //   Since the remainder of each division step < x < 2^32, the computation of
  // each step is now properly contained within uint64_t.
  //   And finally we perform some extra alignment steps for the remaining bits.
  LIBC_INLINE constexpr optional<BigInt<Bits, Signed>>
  div_uint32_times_pow_2(uint32_t x, size_t e) {
    BigInt<Bits, Signed> remainder(0);

    if (x == 0) {
      return nullopt;
    }
    if (e >= Bits) {
      remainder = *this;
      *this = BigInt<Bits, false>(0);
      return remainder;
    }

    BigInt<Bits, Signed> quotient(0);
    uint64_t x64 = static_cast<uint64_t>(x);
    // lower64 = smallest multiple of 64 that is >= e.
    size_t lower64 = ((e >> 6) + ((e & 63) != 0)) << 6;
    // lower_pos is the index of the closest 64-bit chunk >= 2^e.
    size_t lower_pos = lower64 / 64;
    // Keep track of current remainder mod x * 2^(32*i)
    uint64_t rem = 0;
    // pos is the index of the current 64-bit chunk that we are processing.
    size_t pos = WORDCOUNT;

    for (size_t q_pos = WORDCOUNT - lower_pos; q_pos > 0; --q_pos) {
      // q_pos is 1 + the index of the current 64-bit chunk of the quotient
      // being processed.
      // Performing the division / modulus with divisor:
      //   x * 2^(64*q_pos - 32),
      // i.e. using the upper 32-bit of the current 64-bit chunk.
      rem <<= 32;
      rem += val[--pos] >> 32;
      uint64_t q_tmp = rem / x64;
      rem %= x64;

      // Performing the division / modulus with divisor:
      //   x * 2^(64*(q_pos - 1)),
      // i.e. using the lower 32-bit of the current 64-bit chunk.
      rem <<= 32;
      rem += val[pos] & MASK32;
      quotient.val[q_pos - 1] = (q_tmp << 32) + rem / x64;
      rem %= x64;
    }

    // So far, what we have is:
    //   quotient = y / (x * 2^lower64), and
    //        rem = (y % (x * 2^lower64)) / 2^lower64.
    // If (lower64 > e), we will need to perform an extra adjustment of the
    // quotient and remainder, namely:
    //   y / (x * 2^e) = [ y / (x * 2^lower64) ] * 2^(lower64 - e) +
    //                   + (rem * 2^(lower64 - e)) / x
    //   (y % (x * 2^e)) / 2^e = (rem * 2^(lower64 - e)) % x
    size_t last_shift = lower64 - e;

    if (last_shift > 0) {
      // quotient * 2^(lower64 - e)
      quotient <<= last_shift;
      uint64_t q_tmp = 0;
      uint64_t d = val[--pos];
      if (last_shift >= 32) {
        // The shifting (rem * 2^(lower64 - e)) might overflow uint64_t, so we
        // perform a 32-bit shift first.
        rem <<= 32;
        rem += d >> 32;
        d &= MASK32;
        q_tmp = rem / x64;
        rem %= x64;
        last_shift -= 32;
      } else {
        // Only use the upper 32-bit of the current 64-bit chunk.
        d >>= 32;
      }

      if (last_shift > 0) {
        rem <<= 32;
        rem += d;
        q_tmp <<= last_shift;
        x64 <<= 32 - last_shift;
        q_tmp += rem / x64;
        rem %= x64;
      }

      quotient.val[0] += q_tmp;

      if (lower64 - e <= 32) {
        // The remainder rem * 2^(lower64 - e) might overflow to the higher
        // 64-bit chunk.
        if (pos < WORDCOUNT - 1) {
          remainder[pos + 1] = rem >> 32;
        }
        remainder[pos] = (rem << 32) + (val[pos] & MASK32);
      } else {
        remainder[pos] = rem;
      }

    } else {
      remainder[pos] = rem;
    }

    // Set the remaining lower bits of the remainder.
    for (; pos > 0; --pos) {
      remainder[pos - 1] = val[pos - 1];
    }

    *this = quotient;
    return remainder;
  }

  LIBC_INLINE constexpr BigInt<Bits, Signed>
  operator/(const BigInt<Bits, Signed> &other) const {
    BigInt<Bits, Signed> result(*this);
    result.div(other);
    return result;
  }

  LIBC_INLINE constexpr BigInt<Bits, Signed> &
  operator/=(const BigInt<Bits, Signed> &other) {
    div(other);
    return *this;
  }

  LIBC_INLINE constexpr BigInt<Bits, Signed>
  operator%(const BigInt<Bits, Signed> &other) const {
    BigInt<Bits, Signed> result(*this);
    return *result.div(other);
  }

  LIBC_INLINE constexpr BigInt<Bits, Signed> &
  operator*=(const BigInt<Bits, Signed> &other) {
    *this = *this * other;
    return *this;
  }

  LIBC_INLINE constexpr uint64_t clz() {
    uint64_t leading_zeroes = 0;
    for (size_t i = WORDCOUNT; i > 0; --i) {
      if (val[i - 1] == 0) {
        leading_zeroes += sizeof(uint64_t) * 8;
      } else {
        leading_zeroes += countl_zero(val[i - 1]);
        break;
      }
    }
    return leading_zeroes;
  }

  LIBC_INLINE constexpr void shift_left(size_t s) {
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
#endif // __SIZEOF_INT128__
    if (LIBC_UNLIKELY(s == 0))
      return;

    const size_t drop = s / 64;  // Number of words to drop
    const size_t shift = s % 64; // Bits to shift in the remaining words.
    size_t i = WORDCOUNT;

    if (drop < WORDCOUNT) {
      i = WORDCOUNT - 1;
      if (shift > 0) {
        for (size_t j = WORDCOUNT - 1 - drop; j > 0; --i, --j) {
          val[i] = (val[j] << shift) | (val[j - 1] >> (64 - shift));
        }
        val[i] = val[0] << shift;
      } else {
        for (size_t j = WORDCOUNT - 1 - drop; j > 0; --i, --j) {
          val[i] = val[j];
        }
        val[i] = val[0];
      }
    }

    for (size_t j = 0; j < i; ++j) {
      val[j] = 0;
    }
  }

  LIBC_INLINE constexpr BigInt<Bits, Signed> operator<<(size_t s) const {
    BigInt<Bits, Signed> result(*this);
    result.shift_left(s);
    return result;
  }

  LIBC_INLINE constexpr BigInt<Bits, Signed> &operator<<=(size_t s) {
    shift_left(s);
    return *this;
  }

  LIBC_INLINE constexpr void shift_right(size_t s) {
#ifdef __SIZEOF_INT128__
    if constexpr (Bits == 128) {
      // Use builtin 128 bits if available;
      if (s >= 128) {
        val[0] = 0;
        val[1] = 0;
        return;
      }
      __uint128_t tmp = __uint128_t(val[0]) + (__uint128_t(val[1]) << 64);
      if constexpr (Signed) {
        tmp = static_cast<__uint128_t>(static_cast<__int128_t>(tmp) >> s);
      } else {
        tmp >>= s;
      }
      val[0] = uint64_t(tmp);
      val[1] = uint64_t(tmp >> 64);
      return;
    }
#endif // __SIZEOF_INT128__

    if (LIBC_UNLIKELY(s == 0))
      return;
    const size_t drop = s / 64;  // Number of words to drop
    const size_t shift = s % 64; // Bit shift in the remaining words.

    size_t i = 0;
    uint64_t sign = Signed ? (val[WORDCOUNT - 1] >> 63) : 0;

    if (drop < WORDCOUNT) {
      if (shift > 0) {
        for (size_t j = drop; j < WORDCOUNT - 1; ++i, ++j) {
          val[i] = (val[j] >> shift) | (val[j + 1] << (64 - shift));
        }
        if constexpr (Signed) {
          val[i] = static_cast<uint64_t>(
              static_cast<int64_t>(val[WORDCOUNT - 1]) >> shift);
        } else {
          val[i] = val[WORDCOUNT - 1] >> shift;
        }
        ++i;
      } else {
        for (size_t j = drop; j < WORDCOUNT; ++i, ++j) {
          val[i] = val[j];
        }
      }
    }

    for (; i < WORDCOUNT; ++i) {
      val[i] = sign;
    }
  }

  LIBC_INLINE constexpr BigInt<Bits, Signed> operator>>(size_t s) const {
    BigInt<Bits, Signed> result(*this);
    result.shift_right(s);
    return result;
  }

  LIBC_INLINE constexpr BigInt<Bits, Signed> &operator>>=(size_t s) {
    shift_right(s);
    return *this;
  }

  LIBC_INLINE constexpr BigInt<Bits, Signed>
  operator&(const BigInt<Bits, Signed> &other) const {
    BigInt<Bits, Signed> result;
    for (size_t i = 0; i < WORDCOUNT; ++i)
      result.val[i] = val[i] & other.val[i];
    return result;
  }

  LIBC_INLINE constexpr BigInt<Bits, Signed> &
  operator&=(const BigInt<Bits, Signed> &other) {
    for (size_t i = 0; i < WORDCOUNT; ++i)
      val[i] &= other.val[i];
    return *this;
  }

  LIBC_INLINE constexpr BigInt<Bits, Signed>
  operator|(const BigInt<Bits, Signed> &other) const {
    BigInt<Bits, Signed> result;
    for (size_t i = 0; i < WORDCOUNT; ++i)
      result.val[i] = val[i] | other.val[i];
    return result;
  }

  LIBC_INLINE constexpr BigInt<Bits, Signed> &
  operator|=(const BigInt<Bits, Signed> &other) {
    for (size_t i = 0; i < WORDCOUNT; ++i)
      val[i] |= other.val[i];
    return *this;
  }

  LIBC_INLINE constexpr BigInt<Bits, Signed>
  operator^(const BigInt<Bits, Signed> &other) const {
    BigInt<Bits, Signed> result;
    for (size_t i = 0; i < WORDCOUNT; ++i)
      result.val[i] = val[i] ^ other.val[i];
    return result;
  }

  LIBC_INLINE constexpr BigInt<Bits, Signed> &
  operator^=(const BigInt<Bits, Signed> &other) {
    for (size_t i = 0; i < WORDCOUNT; ++i)
      val[i] ^= other.val[i];
    return *this;
  }

  LIBC_INLINE constexpr BigInt<Bits, Signed> operator~() const {
    BigInt<Bits, Signed> result;
    for (size_t i = 0; i < WORDCOUNT; ++i)
      result.val[i] = ~val[i];
    return result;
  }

  LIBC_INLINE constexpr BigInt<Bits, Signed> operator-() const {
    BigInt<Bits, Signed> result = ~(*this);
    result.add(BigInt<Bits, Signed>(1));
    return result;
  }

  LIBC_INLINE constexpr bool
  operator==(const BigInt<Bits, Signed> &other) const {
    for (size_t i = 0; i < WORDCOUNT; ++i) {
      if (val[i] != other.val[i])
        return false;
    }
    return true;
  }

  LIBC_INLINE constexpr bool
  operator!=(const BigInt<Bits, Signed> &other) const {
    for (size_t i = 0; i < WORDCOUNT; ++i) {
      if (val[i] != other.val[i])
        return true;
    }
    return false;
  }

  LIBC_INLINE constexpr bool
  operator>(const BigInt<Bits, Signed> &other) const {
    if constexpr (Signed) {
      // Check for different signs;
      bool a_sign = val[WORDCOUNT - 1] >> 63;
      bool b_sign = other.val[WORDCOUNT - 1] >> 63;
      if (a_sign != b_sign) {
        return b_sign;
      }
    }
    for (size_t i = WORDCOUNT; i > 0; --i) {
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

  LIBC_INLINE constexpr bool
  operator>=(const BigInt<Bits, Signed> &other) const {
    if constexpr (Signed) {
      // Check for different signs;
      bool a_sign = val[WORDCOUNT - 1] >> 63;
      bool b_sign = other.val[WORDCOUNT - 1] >> 63;
      if (a_sign != b_sign) {
        return b_sign;
      }
    }
    for (size_t i = WORDCOUNT; i > 0; --i) {
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

  LIBC_INLINE constexpr bool
  operator<(const BigInt<Bits, Signed> &other) const {
    if constexpr (Signed) {
      // Check for different signs;
      bool a_sign = val[WORDCOUNT - 1] >> 63;
      bool b_sign = other.val[WORDCOUNT - 1] >> 63;
      if (a_sign != b_sign) {
        return a_sign;
      }
    }

    for (size_t i = WORDCOUNT; i > 0; --i) {
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

  LIBC_INLINE constexpr bool
  operator<=(const BigInt<Bits, Signed> &other) const {
    if constexpr (Signed) {
      // Check for different signs;
      bool a_sign = val[WORDCOUNT - 1] >> 63;
      bool b_sign = other.val[WORDCOUNT - 1] >> 63;
      if (a_sign != b_sign) {
        return a_sign;
      }
    }
    for (size_t i = WORDCOUNT; i > 0; --i) {
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

  LIBC_INLINE constexpr BigInt<Bits, Signed> &operator++() {
    BigInt<Bits, Signed> one(1);
    add(one);
    return *this;
  }

  LIBC_INLINE constexpr BigInt<Bits, Signed> operator++(int) {
    BigInt<Bits, Signed> oldval(*this);
    BigInt<Bits, Signed> one(1);
    add(one);
    return oldval;
  }

  LIBC_INLINE constexpr BigInt<Bits, Signed> &operator--() {
    BigInt<Bits, Signed> one(1);
    sub(one);
    return *this;
  }

  LIBC_INLINE constexpr BigInt<Bits, Signed> operator--(int) {
    BigInt<Bits, Signed> oldval(*this);
    BigInt<Bits, Signed> one(1);
    sub(one);
    return oldval;
  }

  // Return the i-th 64-bit word of the number.
  LIBC_INLINE constexpr const uint64_t &operator[](size_t i) const {
    return val[i];
  }

  // Return the i-th 64-bit word of the number.
  LIBC_INLINE constexpr uint64_t &operator[](size_t i) { return val[i]; }

  LIBC_INLINE uint64_t *data() { return val; }

  LIBC_INLINE const uint64_t *data() const { return val; }
};

template <size_t Bits> using UInt = BigInt<Bits, false>;

template <size_t Bits> using Int = BigInt<Bits, true>;

// Provides limits of U/Int<128>.
template <> class numeric_limits<UInt<128>> {
public:
  LIBC_INLINE static constexpr UInt<128> max() {
    return UInt<128>({0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff});
  }
  LIBC_INLINE static constexpr UInt<128> min() { return UInt<128>(0); }
  LIBC_INLINE_VAR static constexpr int digits = 128;
};

template <> class numeric_limits<Int<128>> {
public:
  LIBC_INLINE static constexpr Int<128> max() {
    return Int<128>({0xffff'ffff'ffff'ffff, 0x7fff'ffff'ffff'ffff});
  }
  LIBC_INLINE static constexpr Int<128> min() {
    return Int<128>({0, 0x8000'0000'0000'0000});
  }
  LIBC_INLINE_VAR static constexpr int digits = 128;
};

// Provides is_integral of U/Int<128>, U/Int<192>, U/Int<256>.
template <size_t Bits, bool Signed>
struct is_integral<BigInt<Bits, Signed>> : cpp::true_type {
  static_assert(Bits > 0 && Bits % 64 == 0,
                "Number of bits in BigInt should be a multiple of 64.");
};

// Provides is_unsigned of UInt<128>, UInt<192>, UInt<256>.
template <size_t Bits> struct is_unsigned<UInt<Bits>> : public cpp::true_type {
  static_assert(Bits > 0 && Bits % 64 == 0,
                "Number of bits in UInt should be a multiple of 64.");
};

template <size_t Bits>
struct make_unsigned<Int<Bits>> : type_identity<UInt<Bits>> {
  static_assert(Bits > 0 && Bits % 64 == 0,
                "Number of bits in Int should be a multiple of 64.");
};

template <size_t Bits>
struct make_unsigned<UInt<Bits>> : type_identity<UInt<Bits>> {
  static_assert(Bits > 0 && Bits % 64 == 0,
                "Number of bits in Int should be a multiple of 64.");
};

template <size_t Bits>
struct make_signed<Int<Bits>> : type_identity<Int<Bits>> {
  static_assert(Bits > 0 && Bits % 64 == 0,
                "Number of bits in Int should be a multiple of 64.");
};

template <size_t Bits>
struct make_signed<UInt<Bits>> : type_identity<Int<Bits>> {
  static_assert(Bits > 0 && Bits % 64 == 0,
                "Number of bits in Int should be a multiple of 64.");
};

} // namespace LIBC_NAMESPACE::cpp

#endif // LLVM_LIBC_SRC___SUPPORT_UINT_H
