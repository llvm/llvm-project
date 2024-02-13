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

namespace internal {
template <typename T> struct half_width;

template <> struct half_width<uint64_t> : type_identity<uint32_t> {};
template <> struct half_width<uint32_t> : type_identity<uint16_t> {};
template <> struct half_width<uint16_t> : type_identity<uint8_t> {};
#ifdef __SIZEOF_INT128__
template <> struct half_width<__uint128_t> : type_identity<uint64_t> {};
#endif // __SIZEOF_INT128__

template <typename T> using half_width_t = typename half_width<T>::type;
} // namespace internal

template <size_t Bits, bool Signed, typename WordType = uint64_t>
struct BigInt {
  static_assert(is_integral_v<WordType> && is_unsigned_v<WordType>,
                "WordType must be unsigned integer.");

  LIBC_INLINE_VAR
  static constexpr size_t WORD_SIZE = sizeof(WordType) * CHAR_BIT;

  static_assert(Bits > 0 && Bits % WORD_SIZE == 0,
                "Number of bits in BigInt should be a multiple of WORD_SIZE.");

  LIBC_INLINE_VAR static constexpr size_t WORD_COUNT = Bits / WORD_SIZE;
  cpp::array<WordType, WORD_COUNT> val{};

  LIBC_INLINE constexpr BigInt() = default;

  LIBC_INLINE constexpr BigInt(const BigInt<Bits, Signed, WordType> &other) =
      default;

  template <size_t OtherBits, bool OtherSigned>
  LIBC_INLINE constexpr BigInt(
      const BigInt<OtherBits, OtherSigned, WordType> &other) {
    if (OtherBits >= Bits) {
      for (size_t i = 0; i < WORD_COUNT; ++i)
        val[i] = other[i];
    } else {
      size_t i = 0;
      for (; i < OtherBits / 64; ++i)
        val[i] = other[i];
      WordType sign = 0;
      if constexpr (Signed && OtherSigned) {
        sign = static_cast<WordType>(-static_cast<make_signed_t<WordType>>(
            other[OtherBits / WORD_SIZE - 1] >> (WORD_SIZE - 1)));
      }
      for (; i < WORD_COUNT; ++i)
        val[i] = sign;
    }
  }

  // Construct a BigInt from a C array.
  template <size_t N, enable_if_t<N <= WORD_COUNT, int> = 0>
  LIBC_INLINE constexpr BigInt(const WordType (&nums)[N]) {
    size_t min_wordcount = N < WORD_COUNT ? N : WORD_COUNT;
    size_t i = 0;
    for (; i < min_wordcount; ++i)
      val[i] = nums[i];

    // If nums doesn't completely fill val, then fill the rest with zeroes.
    for (; i < WORD_COUNT; ++i)
      val[i] = 0;
  }

  // Initialize the first word to |v| and the rest to 0.
  template <typename T, typename = cpp::enable_if_t<is_integral_v<T>>>
  LIBC_INLINE constexpr BigInt(T v) {
    val[0] = static_cast<WordType>(v);

    if constexpr (WORD_COUNT == 1)
      return;

    if constexpr (Bits < sizeof(T) * CHAR_BIT) {
      for (int i = 1; i < WORD_COUNT; ++i) {
        v >>= WORD_SIZE;
        val[i] = static_cast<WordType>(v);
      }
      return;
    }

    size_t i = 1;

    if constexpr (WORD_SIZE < sizeof(T) * CHAR_BIT)
      for (; i < sizeof(T) * CHAR_BIT / WORD_SIZE; ++i) {
        v >>= WORD_SIZE;
        val[i] = static_cast<WordType>(v);
      }

    WordType sign = (Signed && (v < 0)) ? ~WordType(0) : WordType(0);
    for (; i < WORD_COUNT; ++i) {
      val[i] = sign;
    }
  }

  LIBC_INLINE constexpr explicit BigInt(
      const cpp::array<WordType, WORD_COUNT> &words) {
    for (size_t i = 0; i < WORD_COUNT; ++i)
      val[i] = words[i];
  }

  template <typename T> LIBC_INLINE constexpr explicit operator T() const {
    return to<T>();
  }

  template <typename T>
  LIBC_INLINE constexpr cpp::enable_if_t<
      cpp::is_integral_v<T> && !cpp::is_same_v<T, bool>, T>
  to() const {
    T lo = static_cast<T>(val[0]);

    constexpr size_t T_BITS = sizeof(T) * CHAR_BIT;

    if constexpr (T_BITS <= WORD_SIZE)
      return lo;

    constexpr size_t MAX_COUNT =
        T_BITS > Bits ? WORD_COUNT : T_BITS / WORD_SIZE;
    for (size_t i = 1; i < MAX_COUNT; ++i)
      lo += static_cast<T>(val[i]) << (WORD_SIZE * i);

    if constexpr (Signed && (T_BITS > Bits)) {
      // Extend sign for negative numbers.
      constexpr T MASK = (~T(0) << Bits);
      if (val[WORD_COUNT - 1] >> (WORD_SIZE - 1))
        lo |= MASK;
    }

    return lo;
  }

  LIBC_INLINE constexpr explicit operator bool() const { return !is_zero(); }

  LIBC_INLINE BigInt<Bits, Signed, WordType> &
  operator=(const BigInt<Bits, Signed, WordType> &other) = default;

  LIBC_INLINE constexpr bool is_zero() const {
    for (size_t i = 0; i < WORD_COUNT; ++i) {
      if (val[i] != 0)
        return false;
    }
    return true;
  }

  // Add x to this number and store the result in this number.
  // Returns the carry value produced by the addition operation.
  LIBC_INLINE constexpr WordType add(const BigInt<Bits, Signed, WordType> &x) {
    SumCarry<WordType> s{0, 0};
    for (size_t i = 0; i < WORD_COUNT; ++i) {
      s = add_with_carry_const(val[i], x.val[i], s.carry);
      val[i] = s.sum;
    }
    return s.carry;
  }

  LIBC_INLINE constexpr BigInt<Bits, Signed, WordType>
  operator+(const BigInt<Bits, Signed, WordType> &other) const {
    BigInt<Bits, Signed, WordType> result;
    SumCarry<WordType> s{0, 0};
    for (size_t i = 0; i < WORD_COUNT; ++i) {
      s = add_with_carry(val[i], other.val[i], s.carry);
      result.val[i] = s.sum;
    }
    return result;
  }

  // This will only apply when initializing a variable from constant values, so
  // it will always use the constexpr version of add_with_carry.
  LIBC_INLINE constexpr BigInt<Bits, Signed, WordType>
  operator+(BigInt<Bits, Signed, WordType> &&other) const {
    BigInt<Bits, Signed, WordType> result;
    SumCarry<WordType> s{0, 0};
    for (size_t i = 0; i < WORD_COUNT; ++i) {
      s = add_with_carry_const(val[i], other.val[i], s.carry);
      result.val[i] = s.sum;
    }
    return result;
  }

  LIBC_INLINE constexpr BigInt<Bits, Signed, WordType> &
  operator+=(const BigInt<Bits, Signed, WordType> &other) {
    add(other); // Returned carry value is ignored.
    return *this;
  }

  // Subtract x to this number and store the result in this number.
  // Returns the carry value produced by the subtraction operation.
  LIBC_INLINE constexpr WordType sub(const BigInt<Bits, Signed, WordType> &x) {
    DiffBorrow<WordType> d{0, 0};
    for (size_t i = 0; i < WORD_COUNT; ++i) {
      d = sub_with_borrow_const(val[i], x.val[i], d.borrow);
      val[i] = d.diff;
    }
    return d.borrow;
  }

  LIBC_INLINE constexpr BigInt<Bits, Signed, WordType>
  operator-(const BigInt<Bits, Signed, WordType> &other) const {
    BigInt<Bits, Signed, WordType> result;
    DiffBorrow<WordType> d{0, 0};
    for (size_t i = 0; i < WORD_COUNT; ++i) {
      d = sub_with_borrow(val[i], other.val[i], d.borrow);
      result.val[i] = d.diff;
    }
    return result;
  }

  LIBC_INLINE constexpr BigInt<Bits, Signed, WordType>
  operator-(BigInt<Bits, Signed, WordType> &&other) const {
    BigInt<Bits, Signed, WordType> result;
    DiffBorrow<WordType> d{0, 0};
    for (size_t i = 0; i < WORD_COUNT; ++i) {
      d = sub_with_borrow_const(val[i], other.val[i], d.borrow);
      result.val[i] = d.diff;
    }
    return result;
  }

  LIBC_INLINE constexpr BigInt<Bits, Signed, WordType> &
  operator-=(const BigInt<Bits, Signed, WordType> &other) {
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
  LIBC_INLINE constexpr WordType mul(WordType x) {
    BigInt<2 * WORD_SIZE, Signed, WordType> partial_sum(0);
    WordType carry = 0;
    for (size_t i = 0; i < WORD_COUNT; ++i) {
      NumberPair<WordType> prod = full_mul(val[i], x);
      BigInt<2 * WORD_SIZE, Signed, WordType> tmp({prod.lo, prod.hi});
      carry += partial_sum.add(tmp);
      val[i] = partial_sum.val[0];
      partial_sum.val[0] = partial_sum.val[1];
      partial_sum.val[1] = carry;
      carry = 0;
    }
    return partial_sum.val[1];
  }

  LIBC_INLINE constexpr BigInt<Bits, Signed, WordType>
  operator*(const BigInt<Bits, Signed, WordType> &other) const {
    if constexpr (Signed) {
      BigInt<Bits, false, WordType> a(*this);
      BigInt<Bits, false, WordType> b(other);
      bool a_neg = (a.val[WORD_COUNT - 1] >> (WORD_SIZE - 1));
      bool b_neg = (b.val[WORD_COUNT - 1] >> (WORD_SIZE - 1));
      if (a_neg)
        a = -a;
      if (b_neg)
        b = -b;
      BigInt<Bits, false, WordType> prod = a * b;
      if (a_neg != b_neg)
        prod = -prod;
      return static_cast<BigInt<Bits, true, WordType>>(prod);
    } else {

      if constexpr (WORD_COUNT == 1) {
        return {val[0] * other.val[0]};
      } else {
        BigInt<Bits, Signed, WordType> result(0);
        BigInt<2 * WORD_SIZE, Signed, WordType> partial_sum(0);
        WordType carry = 0;
        for (size_t i = 0; i < WORD_COUNT; ++i) {
          for (size_t j = 0; j <= i; j++) {
            NumberPair<WordType> prod = full_mul(val[j], other.val[i - j]);
            BigInt<2 * WORD_SIZE, Signed, WordType> tmp({prod.lo, prod.hi});
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
  LIBC_INLINE constexpr BigInt<Bits + OtherBits, Signed, WordType>
  ful_mul(const BigInt<OtherBits, Signed, WordType> &other) const {
    BigInt<Bits + OtherBits, Signed, WordType> result(0);
    BigInt<2 * WORD_SIZE, Signed, WordType> partial_sum(0);
    WordType carry = 0;
    constexpr size_t OTHER_WORDCOUNT =
        BigInt<OtherBits, Signed, WordType>::WORD_COUNT;
    for (size_t i = 0; i <= WORD_COUNT + OTHER_WORDCOUNT - 2; ++i) {
      const size_t lower_idx =
          i < OTHER_WORDCOUNT ? 0 : i - OTHER_WORDCOUNT + 1;
      const size_t upper_idx = i < WORD_COUNT ? i : WORD_COUNT - 1;
      for (size_t j = lower_idx; j <= upper_idx; ++j) {
        NumberPair<WordType> prod = full_mul(val[j], other.val[i - j]);
        BigInt<2 * WORD_SIZE, Signed, WordType> tmp({prod.lo, prod.hi});
        carry += partial_sum.add(tmp);
      }
      result.val[i] = partial_sum.val[0];
      partial_sum.val[0] = partial_sum.val[1];
      partial_sum.val[1] = carry;
      carry = 0;
    }
    result.val[WORD_COUNT + OTHER_WORDCOUNT - 1] = partial_sum.val[0];
    return result;
  }

  // Fast hi part of the full product.  The normal product `operator*` returns
  // `Bits` least significant bits of the full product, while this function will
  // approximate `Bits` most significant bits of the full product with errors
  // bounded by:
  //   0 <= (a.full_mul(b) >> Bits) - a.quick_mul_hi(b)) <= WORD_COUNT - 1.
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
  //   Bits  WORD_COUNT  ful_mul  quick_mul_hi  Error bound
  //    128      2         4           3            1
  //    196      3         9           6            2
  //    256      4        16          10            3
  //    512      8        64          36            7
  LIBC_INLINE constexpr BigInt<Bits, Signed, WordType>
  quick_mul_hi(const BigInt<Bits, Signed, WordType> &other) const {
    BigInt<Bits, Signed, WordType> result(0);
    BigInt<2 * WORD_SIZE, Signed, WordType> partial_sum(0);
    WordType carry = 0;
    // First round of accumulation for those at WORD_COUNT - 1 in the full
    // product.
    for (size_t i = 0; i < WORD_COUNT; ++i) {
      NumberPair<WordType> prod =
          full_mul(val[i], other.val[WORD_COUNT - 1 - i]);
      BigInt<2 * WORD_SIZE, Signed, WordType> tmp({prod.lo, prod.hi});
      carry += partial_sum.add(tmp);
    }
    for (size_t i = WORD_COUNT; i < 2 * WORD_COUNT - 1; ++i) {
      partial_sum.val[0] = partial_sum.val[1];
      partial_sum.val[1] = carry;
      carry = 0;
      for (size_t j = i - WORD_COUNT + 1; j < WORD_COUNT; ++j) {
        NumberPair<WordType> prod = full_mul(val[j], other.val[i - j]);
        BigInt<2 * WORD_SIZE, Signed, WordType> tmp({prod.lo, prod.hi});
        carry += partial_sum.add(tmp);
      }
      result.val[i - WORD_COUNT] = partial_sum.val[0];
    }
    result.val[WORD_COUNT - 1] = partial_sum.val[1];
    return result;
  }

  // pow takes a power and sets this to its starting value to that power. Zero
  // to the zeroth power returns 1.
  LIBC_INLINE constexpr void pow_n(uint64_t power) {
    BigInt<Bits, Signed, WordType> result = 1;
    BigInt<Bits, Signed, WordType> cur_power = *this;

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
  LIBC_INLINE constexpr optional<BigInt<Bits, Signed, WordType>>
  div(const BigInt<Bits, Signed, WordType> &other) {
    BigInt<Bits, Signed, WordType> remainder(0);
    if (*this < other) {
      remainder = *this;
      *this = BigInt<Bits, Signed, WordType>(0);
      return remainder;
    }
    if (other == 1) {
      return remainder;
    }
    if (other == 0) {
      return nullopt;
    }

    BigInt<Bits, Signed, WordType> quotient(0);
    BigInt<Bits, Signed, WordType> subtractor = other;
    int cur_bit = static_cast<int>(subtractor.clz() - this->clz());
    subtractor.shift_left(cur_bit);

    for (; cur_bit >= 0 && *this > 0; --cur_bit, subtractor.shift_right(1)) {
      if (*this >= subtractor) {
        this->sub(subtractor);
        quotient = quotient | (BigInt<Bits, Signed, WordType>(1) << cur_bit);
      }
    }
    remainder = *this;
    *this = quotient;
    return remainder;
  }

  // Efficiently perform BigInt / (x * 2^e), where x is a half-word-size
  // unsigned integer, and return the remainder. The main idea is as follow:
  //   Let q = y / (x * 2^e) be the quotient, and
  //       r = y % (x * 2^e) be the remainder.
  //   First, notice that:
  //     r % (2^e) = y % (2^e),
  // so we just need to focus on all the bits of y that is >= 2^e.
  //   To speed up the shift-and-add steps, we only use x as the divisor, and
  // performing 32-bit shiftings instead of bit-by-bit shiftings.
  //   Since the remainder of each division step < x < 2^(WORD_SIZE / 2), the
  // computation of each step is now properly contained within WordType.
  //   And finally we perform some extra alignment steps for the remaining bits.
  LIBC_INLINE constexpr optional<BigInt<Bits, Signed, WordType>>
  div_uint_half_times_pow_2(internal::half_width_t<WordType> x, size_t e) {
    BigInt<Bits, Signed, WordType> remainder(0);

    if (x == 0) {
      return nullopt;
    }
    if (e >= Bits) {
      remainder = *this;
      *this = BigInt<Bits, false, WordType>(0);
      return remainder;
    }

    BigInt<Bits, Signed, WordType> quotient(0);
    WordType x_word = static_cast<WordType>(x);
    constexpr size_t LOG2_WORD_SIZE = bit_width(WORD_SIZE) - 1;
    constexpr size_t HALF_WORD_SIZE = WORD_SIZE >> 1;
    constexpr WordType HALF_MASK = ((WordType(1) << HALF_WORD_SIZE) - 1);
    // lower = smallest multiple of WORD_SIZE that is >= e.
    size_t lower = ((e >> LOG2_WORD_SIZE) + ((e & (WORD_SIZE - 1)) != 0))
                   << LOG2_WORD_SIZE;
    // lower_pos is the index of the closest WORD_SIZE-bit chunk >= 2^e.
    size_t lower_pos = lower / WORD_SIZE;
    // Keep track of current remainder mod x * 2^(32*i)
    WordType rem = 0;
    // pos is the index of the current 64-bit chunk that we are processing.
    size_t pos = WORD_COUNT;

    // TODO: look into if constexpr(Bits > 256) skip leading zeroes.

    for (size_t q_pos = WORD_COUNT - lower_pos; q_pos > 0; --q_pos) {
      // q_pos is 1 + the index of the current WORD_SIZE-bit chunk of the
      // quotient being processed. Performing the division / modulus with
      // divisor:
      //   x * 2^(WORD_SIZE*q_pos - WORD_SIZE/2),
      // i.e. using the upper (WORD_SIZE/2)-bit of the current WORD_SIZE-bit
      // chunk.
      rem <<= HALF_WORD_SIZE;
      rem += val[--pos] >> HALF_WORD_SIZE;
      WordType q_tmp = rem / x_word;
      rem %= x_word;

      // Performing the division / modulus with divisor:
      //   x * 2^(WORD_SIZE*(q_pos - 1)),
      // i.e. using the lower (WORD_SIZE/2)-bit of the current WORD_SIZE-bit
      // chunk.
      rem <<= HALF_WORD_SIZE;
      rem += val[pos] & HALF_MASK;
      quotient.val[q_pos - 1] = (q_tmp << HALF_WORD_SIZE) + rem / x_word;
      rem %= x_word;
    }

    // So far, what we have is:
    //   quotient = y / (x * 2^lower), and
    //        rem = (y % (x * 2^lower)) / 2^lower.
    // If (lower > e), we will need to perform an extra adjustment of the
    // quotient and remainder, namely:
    //   y / (x * 2^e) = [ y / (x * 2^lower) ] * 2^(lower - e) +
    //                   + (rem * 2^(lower - e)) / x
    //   (y % (x * 2^e)) / 2^e = (rem * 2^(lower - e)) % x
    size_t last_shift = lower - e;

    if (last_shift > 0) {
      // quotient * 2^(lower - e)
      quotient <<= last_shift;
      WordType q_tmp = 0;
      WordType d = val[--pos];
      if (last_shift >= HALF_WORD_SIZE) {
        // The shifting (rem * 2^(lower - e)) might overflow WordTyoe, so we
        // perform a HALF_WORD_SIZE-bit shift first.
        rem <<= HALF_WORD_SIZE;
        rem += d >> HALF_WORD_SIZE;
        d &= HALF_MASK;
        q_tmp = rem / x_word;
        rem %= x_word;
        last_shift -= HALF_WORD_SIZE;
      } else {
        // Only use the upper HALF_WORD_SIZE-bit of the current WORD_SIZE-bit
        // chunk.
        d >>= HALF_WORD_SIZE;
      }

      if (last_shift > 0) {
        rem <<= HALF_WORD_SIZE;
        rem += d;
        q_tmp <<= last_shift;
        x_word <<= HALF_WORD_SIZE - last_shift;
        q_tmp += rem / x_word;
        rem %= x_word;
      }

      quotient.val[0] += q_tmp;

      if (lower - e <= HALF_WORD_SIZE) {
        // The remainder rem * 2^(lower - e) might overflow to the higher
        // WORD_SIZE-bit chunk.
        if (pos < WORD_COUNT - 1) {
          remainder[pos + 1] = rem >> HALF_WORD_SIZE;
        }
        remainder[pos] = (rem << HALF_WORD_SIZE) + (val[pos] & HALF_MASK);
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

  LIBC_INLINE constexpr BigInt<Bits, Signed, WordType>
  operator/(const BigInt<Bits, Signed, WordType> &other) const {
    BigInt<Bits, Signed, WordType> result(*this);
    result.div(other);
    return result;
  }

  LIBC_INLINE constexpr BigInt<Bits, Signed, WordType> &
  operator/=(const BigInt<Bits, Signed, WordType> &other) {
    div(other);
    return *this;
  }

  LIBC_INLINE constexpr BigInt<Bits, Signed, WordType>
  operator%(const BigInt<Bits, Signed, WordType> &other) const {
    BigInt<Bits, Signed, WordType> result(*this);
    return *result.div(other);
  }

  LIBC_INLINE constexpr BigInt<Bits, Signed, WordType> &
  operator*=(const BigInt<Bits, Signed, WordType> &other) {
    *this = *this * other;
    return *this;
  }

  LIBC_INLINE constexpr uint64_t clz() {
    uint64_t leading_zeroes = 0;
    for (size_t i = WORD_COUNT; i > 0; --i) {
      if (val[i - 1] == 0) {
        leading_zeroes += WORD_SIZE;
      } else {
        leading_zeroes += countl_zero(val[i - 1]);
        break;
      }
    }
    return leading_zeroes;
  }

  LIBC_INLINE constexpr void shift_left(size_t s) {
    if constexpr (Bits == WORD_SIZE) {
      // Use native types if possible.
      if (s >= WORD_SIZE) {
        val[0] = 0;
        return;
      }
      val[0] <<= s;
      return;
    }
    if constexpr ((Bits == 64) && (WORD_SIZE == 32)) {
      // Use builtin 64 bits for 32-bit base type if available;
      if (s >= 64) {
        val[0] = 0;
        val[1] = 0;
        return;
      }
      uint64_t tmp = uint64__t(val[0]) + (uint64_t(val[1]) << 62);
      tmp <<= s;
      val[0] = uint32_t(tmp);
      val[1] = uint32_t(tmp >> 32);
      return;
    }
#ifdef __SIZEOF_INT128__
    if constexpr ((Bits == 128) && (WORD_SIZE == 64)) {
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

    const size_t drop = s / WORD_SIZE;  // Number of words to drop
    const size_t shift = s % WORD_SIZE; // Bits to shift in the remaining words.
    size_t i = WORD_COUNT;

    if (drop < WORD_COUNT) {
      i = WORD_COUNT - 1;
      if (shift > 0) {
        for (size_t j = WORD_COUNT - 1 - drop; j > 0; --i, --j) {
          val[i] = (val[j] << shift) | (val[j - 1] >> (WORD_SIZE - shift));
        }
        val[i] = val[0] << shift;
      } else {
        for (size_t j = WORD_COUNT - 1 - drop; j > 0; --i, --j) {
          val[i] = val[j];
        }
        val[i] = val[0];
      }
    }

    for (size_t j = 0; j < i; ++j) {
      val[j] = 0;
    }
  }

  LIBC_INLINE constexpr BigInt<Bits, Signed, WordType>
  operator<<(size_t s) const {
    BigInt<Bits, Signed, WordType> result(*this);
    result.shift_left(s);
    return result;
  }

  LIBC_INLINE constexpr BigInt<Bits, Signed, WordType> &operator<<=(size_t s) {
    shift_left(s);
    return *this;
  }

  LIBC_INLINE constexpr void shift_right(size_t s) {
    if constexpr ((Bits == 64) && (WORD_SIZE == 32)) {
      // Use builtin 64 bits if available;
      if (s >= 64) {
        val[0] = 0;
        val[1] = 0;
        return;
      }
      uint64_t tmp = uint64_t(val[0]) + (uint64_t(val[1]) << 32);
      if constexpr (Signed) {
        tmp = static_cast<uint64_t>(static_cast<int64_t>(tmp) >> s);
      } else {
        tmp >>= s;
      }
      val[0] = uint32_t(tmp);
      val[1] = uint32_t(tmp >> 32);
      return;
    }
#ifdef __SIZEOF_INT128__
    if constexpr ((Bits == 128) && (WORD_SIZE == 64)) {
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
    const size_t drop = s / WORD_SIZE;  // Number of words to drop
    const size_t shift = s % WORD_SIZE; // Bit shift in the remaining words.

    size_t i = 0;
    WordType sign = Signed ? (val[WORD_COUNT - 1] >> (WORD_SIZE - 1)) : 0;

    if (drop < WORD_COUNT) {
      if (shift > 0) {
        for (size_t j = drop; j < WORD_COUNT - 1; ++i, ++j) {
          val[i] = (val[j] >> shift) | (val[j + 1] << (WORD_SIZE - shift));
        }
        if constexpr (Signed) {
          val[i] = static_cast<WordType>(
              static_cast<cpp::make_signed_t<WordType>>(val[WORD_COUNT - 1]) >>
              shift);
        } else {
          val[i] = val[WORD_COUNT - 1] >> shift;
        }
        ++i;
      } else {
        for (size_t j = drop; j < WORD_COUNT; ++i, ++j) {
          val[i] = val[j];
        }
      }
    }

    for (; i < WORD_COUNT; ++i) {
      val[i] = sign;
    }
  }

  LIBC_INLINE constexpr BigInt<Bits, Signed, WordType>
  operator>>(size_t s) const {
    BigInt<Bits, Signed, WordType> result(*this);
    result.shift_right(s);
    return result;
  }

  LIBC_INLINE constexpr BigInt<Bits, Signed, WordType> &operator>>=(size_t s) {
    shift_right(s);
    return *this;
  }

  LIBC_INLINE constexpr BigInt<Bits, Signed, WordType>
  operator&(const BigInt<Bits, Signed, WordType> &other) const {
    BigInt<Bits, Signed, WordType> result;
    for (size_t i = 0; i < WORD_COUNT; ++i)
      result.val[i] = val[i] & other.val[i];
    return result;
  }

  LIBC_INLINE constexpr BigInt<Bits, Signed, WordType> &
  operator&=(const BigInt<Bits, Signed, WordType> &other) {
    for (size_t i = 0; i < WORD_COUNT; ++i)
      val[i] &= other.val[i];
    return *this;
  }

  LIBC_INLINE constexpr BigInt<Bits, Signed, WordType>
  operator|(const BigInt<Bits, Signed, WordType> &other) const {
    BigInt<Bits, Signed, WordType> result;
    for (size_t i = 0; i < WORD_COUNT; ++i)
      result.val[i] = val[i] | other.val[i];
    return result;
  }

  LIBC_INLINE constexpr BigInt<Bits, Signed, WordType> &
  operator|=(const BigInt<Bits, Signed, WordType> &other) {
    for (size_t i = 0; i < WORD_COUNT; ++i)
      val[i] |= other.val[i];
    return *this;
  }

  LIBC_INLINE constexpr BigInt<Bits, Signed, WordType>
  operator^(const BigInt<Bits, Signed, WordType> &other) const {
    BigInt<Bits, Signed, WordType> result;
    for (size_t i = 0; i < WORD_COUNT; ++i)
      result.val[i] = val[i] ^ other.val[i];
    return result;
  }

  LIBC_INLINE constexpr BigInt<Bits, Signed, WordType> &
  operator^=(const BigInt<Bits, Signed, WordType> &other) {
    for (size_t i = 0; i < WORD_COUNT; ++i)
      val[i] ^= other.val[i];
    return *this;
  }

  LIBC_INLINE constexpr BigInt<Bits, Signed, WordType> operator~() const {
    BigInt<Bits, Signed, WordType> result;
    for (size_t i = 0; i < WORD_COUNT; ++i)
      result.val[i] = ~val[i];
    return result;
  }

  LIBC_INLINE constexpr BigInt<Bits, Signed, WordType> operator-() const {
    BigInt<Bits, Signed, WordType> result = ~(*this);
    result.add(BigInt<Bits, Signed, WordType>(1));
    return result;
  }

  LIBC_INLINE constexpr bool
  operator==(const BigInt<Bits, Signed, WordType> &other) const {
    for (size_t i = 0; i < WORD_COUNT; ++i) {
      if (val[i] != other.val[i])
        return false;
    }
    return true;
  }

  LIBC_INLINE constexpr bool
  operator!=(const BigInt<Bits, Signed, WordType> &other) const {
    for (size_t i = 0; i < WORD_COUNT; ++i) {
      if (val[i] != other.val[i])
        return true;
    }
    return false;
  }

  LIBC_INLINE constexpr bool
  operator>(const BigInt<Bits, Signed, WordType> &other) const {
    if constexpr (Signed) {
      // Check for different signs;
      bool a_sign = val[WORD_COUNT - 1] >> (WORD_SIZE - 1);
      bool b_sign = other.val[WORD_COUNT - 1] >> (WORD_SIZE - 1);
      if (a_sign != b_sign) {
        return b_sign;
      }
    }
    for (size_t i = WORD_COUNT; i > 0; --i) {
      WordType word = val[i - 1];
      WordType other_word = other.val[i - 1];
      if (word > other_word)
        return true;
      else if (word < other_word)
        return false;
    }
    // Equal
    return false;
  }

  LIBC_INLINE constexpr bool
  operator>=(const BigInt<Bits, Signed, WordType> &other) const {
    if constexpr (Signed) {
      // Check for different signs;
      bool a_sign = val[WORD_COUNT - 1] >> (WORD_SIZE - 1);
      bool b_sign = other.val[WORD_COUNT - 1] >> (WORD_SIZE - 1);
      if (a_sign != b_sign) {
        return b_sign;
      }
    }
    for (size_t i = WORD_COUNT; i > 0; --i) {
      WordType word = val[i - 1];
      WordType other_word = other.val[i - 1];
      if (word > other_word)
        return true;
      else if (word < other_word)
        return false;
    }
    // Equal
    return true;
  }

  LIBC_INLINE constexpr bool
  operator<(const BigInt<Bits, Signed, WordType> &other) const {
    if constexpr (Signed) {
      // Check for different signs;
      bool a_sign = val[WORD_COUNT - 1] >> (WORD_SIZE - 1);
      bool b_sign = other.val[WORD_COUNT - 1] >> (WORD_SIZE - 1);
      if (a_sign != b_sign) {
        return a_sign;
      }
    }

    for (size_t i = WORD_COUNT; i > 0; --i) {
      WordType word = val[i - 1];
      WordType other_word = other.val[i - 1];
      if (word > other_word)
        return false;
      else if (word < other_word)
        return true;
    }
    // Equal
    return false;
  }

  LIBC_INLINE constexpr bool
  operator<=(const BigInt<Bits, Signed, WordType> &other) const {
    if constexpr (Signed) {
      // Check for different signs;
      bool a_sign = val[WORD_COUNT - 1] >> (WORD_SIZE - 1);
      bool b_sign = other.val[WORD_COUNT - 1] >> (WORD_SIZE - 1);
      if (a_sign != b_sign) {
        return a_sign;
      }
    }
    for (size_t i = WORD_COUNT; i > 0; --i) {
      WordType word = val[i - 1];
      WordType other_word = other.val[i - 1];
      if (word > other_word)
        return false;
      else if (word < other_word)
        return true;
    }
    // Equal
    return true;
  }

  LIBC_INLINE constexpr BigInt<Bits, Signed, WordType> &operator++() {
    BigInt<Bits, Signed, WordType> one(1);
    add(one);
    return *this;
  }

  LIBC_INLINE constexpr BigInt<Bits, Signed, WordType> operator++(int) {
    BigInt<Bits, Signed, WordType> oldval(*this);
    BigInt<Bits, Signed, WordType> one(1);
    add(one);
    return oldval;
  }

  LIBC_INLINE constexpr BigInt<Bits, Signed, WordType> &operator--() {
    BigInt<Bits, Signed, WordType> one(1);
    sub(one);
    return *this;
  }

  LIBC_INLINE constexpr BigInt<Bits, Signed, WordType> operator--(int) {
    BigInt<Bits, Signed, WordType> oldval(*this);
    BigInt<Bits, Signed, WordType> one(1);
    sub(one);
    return oldval;
  }

  // Return the i-th 64-bit word of the number.
  LIBC_INLINE constexpr const WordType &operator[](size_t i) const {
    return val[i];
  }

  // Return the i-th 64-bit word of the number.
  LIBC_INLINE constexpr WordType &operator[](size_t i) { return val[i]; }

  LIBC_INLINE WordType *data() { return val; }

  LIBC_INLINE const WordType *data() const { return val; }
};

template <size_t Bits>
using UInt =
    typename cpp::conditional_t<Bits == 32, BigInt<32, false, uint32_t>,
                                BigInt<Bits, false, uint64_t>>;

template <size_t Bits>
using Int = typename cpp::conditional_t<Bits == 32, BigInt<32, true, uint32_t>,
                                        BigInt<Bits, true, uint64_t>>;

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
template <size_t Bits, bool Signed, typename T>
struct is_integral<BigInt<Bits, Signed, T>> : cpp::true_type {};

// Provides is_unsigned of UInt<128>, UInt<192>, UInt<256>.
template <size_t Bits, bool Signed, typename T>
struct is_unsigned<BigInt<Bits, Signed, T>> : cpp::bool_constant<!Signed> {};

template <size_t Bits, bool Signed, typename T>
struct make_unsigned<BigInt<Bits, Signed, T>>
    : type_identity<BigInt<Bits, false, T>> {};

template <size_t Bits, bool Signed, typename T>
struct make_signed<BigInt<Bits, Signed, T>>
    : type_identity<BigInt<Bits, true, T>> {};

namespace internal {
template <typename T> struct is_custom_uint : cpp::false_type {};

template <size_t Bits, bool Signed, typename T>
struct is_custom_uint<BigInt<Bits, Signed, T>> : cpp::true_type {};
} // namespace internal

// bit_cast to UInt
// Note: The standard scheme for SFINAE selection is to have exactly one
// function instanciation valid at a time. This is usually done by having a
// predicate in one function and the negated predicate in the other one.
// e.g.
// template<typename = cpp::enable_if_t< is_custom_uint<To>::value == true> ...
// template<typename = cpp::enable_if_t< is_custom_uint<To>::value == false> ...
//
// Unfortunately this would make the default 'cpp::bit_cast' aware of
// 'is_custom_uint' (or any other customization). To prevent exposing all
// customizations in the original function, we create a different function with
// four 'typename's instead of three - otherwise it would be considered as a
// redeclaration of the same function leading to "error: template parameter
// redefines default argument".
template <typename To, typename From,
          typename = cpp::enable_if_t<sizeof(To) == sizeof(From) &&
                                      cpp::is_trivially_copyable<To>::value &&
                                      cpp::is_trivially_copyable<From>::value>,
          typename = cpp::enable_if_t<internal::is_custom_uint<To>::value>>
LIBC_INLINE constexpr To bit_cast(const From &from) {
  To out;
  using Storage = decltype(out.val);
  out.val = cpp::bit_cast<Storage>(from);
  return out;
}

// bit_cast from UInt
template <
    typename To, size_t Bits,
    typename = cpp::enable_if_t<sizeof(To) == sizeof(UInt<Bits>) &&
                                cpp::is_trivially_constructible<To>::value &&
                                cpp::is_trivially_copyable<To>::value &&
                                cpp::is_trivially_copyable<UInt<Bits>>::value>>
LIBC_INLINE constexpr To bit_cast(const UInt<Bits> &from) {
  return cpp::bit_cast<To>(from.val);
}

} // namespace LIBC_NAMESPACE::cpp

#endif // LLVM_LIBC_SRC___SUPPORT_UINT_H
