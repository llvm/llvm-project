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
#include "src/__support/macros/attributes.h"   // LIBC_INLINE
#include "src/__support/macros/optimization.h" // LIBC_UNLIKELY
#include "src/__support/macros/properties/types.h" // LIBC_TYPES_HAS_INT128, LIBC_TYPES_HAS_INT64
#include "src/__support/math_extras.h" // SumCarry, DiffBorrow
#include "src/__support/number_pair.h"

#include <stddef.h> // For size_t
#include <stdint.h>

namespace LIBC_NAMESPACE {

namespace internal {
template <typename T> struct half_width;

template <> struct half_width<uint64_t> : cpp::type_identity<uint32_t> {};
template <> struct half_width<uint32_t> : cpp::type_identity<uint16_t> {};
template <> struct half_width<uint16_t> : cpp::type_identity<uint8_t> {};
#ifdef LIBC_TYPES_HAS_INT128
template <> struct half_width<__uint128_t> : cpp::type_identity<uint64_t> {};
#endif // LIBC_TYPES_HAS_INT128

template <typename T> using half_width_t = typename half_width<T>::type;

template <typename T> constexpr NumberPair<T> full_mul(T a, T b) {
  NumberPair<T> pa = split(a);
  NumberPair<T> pb = split(b);
  NumberPair<T> prod;

  prod.lo = pa.lo * pb.lo;                    // exact
  prod.hi = pa.hi * pb.hi;                    // exact
  NumberPair<T> lo_hi = split(pa.lo * pb.hi); // exact
  NumberPair<T> hi_lo = split(pa.hi * pb.lo); // exact

  constexpr size_t HALF_BIT_WIDTH = sizeof(T) * CHAR_BIT / 2;

  auto r1 = add_with_carry(prod.lo, lo_hi.lo << HALF_BIT_WIDTH, T(0));
  prod.lo = r1.sum;
  prod.hi = add_with_carry(prod.hi, lo_hi.hi, r1.carry).sum;

  auto r2 = add_with_carry(prod.lo, hi_lo.lo << HALF_BIT_WIDTH, T(0));
  prod.lo = r2.sum;
  prod.hi = add_with_carry(prod.hi, hi_lo.hi, r2.carry).sum;

  return prod;
}

template <>
LIBC_INLINE constexpr NumberPair<uint32_t> full_mul<uint32_t>(uint32_t a,
                                                              uint32_t b) {
  uint64_t prod = uint64_t(a) * uint64_t(b);
  NumberPair<uint32_t> result;
  result.lo = uint32_t(prod);
  result.hi = uint32_t(prod >> 32);
  return result;
}

#ifdef LIBC_TYPES_HAS_INT128
template <>
LIBC_INLINE constexpr NumberPair<uint64_t> full_mul<uint64_t>(uint64_t a,
                                                              uint64_t b) {
  __uint128_t prod = __uint128_t(a) * __uint128_t(b);
  NumberPair<uint64_t> result;
  result.lo = uint64_t(prod);
  result.hi = uint64_t(prod >> 64);
  return result;
}
#endif // LIBC_TYPES_HAS_INT128

} // namespace internal

template <size_t Bits, bool Signed, typename WordType = uint64_t>
struct BigInt {
  static_assert(cpp::is_integral_v<WordType> && cpp::is_unsigned_v<WordType>,
                "WordType must be unsigned integer.");

  using word_type = WordType;
  LIBC_INLINE_VAR static constexpr bool SIGNED = Signed;
  LIBC_INLINE_VAR static constexpr size_t BITS = Bits;
  LIBC_INLINE_VAR
  static constexpr size_t WORD_SIZE = sizeof(WordType) * CHAR_BIT;

  static_assert(Bits > 0 && Bits % WORD_SIZE == 0,
                "Number of bits in BigInt should be a multiple of WORD_SIZE.");

  LIBC_INLINE_VAR static constexpr size_t WORD_COUNT = Bits / WORD_SIZE;

  using unsigned_type = BigInt<BITS, false, word_type>;
  using signed_type = BigInt<BITS, true, word_type>;

  cpp::array<WordType, WORD_COUNT> val{};

  LIBC_INLINE constexpr BigInt() = default;

  LIBC_INLINE constexpr BigInt(const BigInt &other) = default;

  template <size_t OtherBits, bool OtherSigned>
  LIBC_INLINE constexpr BigInt(
      const BigInt<OtherBits, OtherSigned, WordType> &other) {
    if (OtherBits >= Bits) {
      for (size_t i = 0; i < WORD_COUNT; ++i)
        val[i] = other[i];
    } else {
      size_t i = 0;
      for (; i < OtherBits / WORD_SIZE; ++i)
        val[i] = other[i];
      WordType sign = 0;
      if constexpr (Signed && OtherSigned) {
        sign = static_cast<WordType>(
            -static_cast<cpp::make_signed_t<WordType>>(other.is_neg()));
      }
      for (; i < WORD_COUNT; ++i)
        val[i] = sign;
    }
  }

  // Construct a BigInt from a C array.
  template <size_t N, cpp::enable_if_t<N <= WORD_COUNT, int> = 0>
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
  template <typename T, typename = cpp::enable_if_t<cpp::is_integral_v<T>>>
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

  // TODO: Reuse the Sign type.
  LIBC_INLINE constexpr bool is_neg() const {
    return val.back() >> (WORD_SIZE - 1);
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
      if (is_neg())
        lo |= MASK;
    }

    return lo;
  }

  LIBC_INLINE constexpr explicit operator bool() const { return !is_zero(); }

  LIBC_INLINE constexpr BigInt &operator=(const BigInt &other) = default;

  LIBC_INLINE constexpr bool is_zero() const {
    for (size_t i = 0; i < WORD_COUNT; ++i) {
      if (val[i] != 0)
        return false;
    }
    return true;
  }

  // Add x to this number and store the result in this number.
  // Returns the carry value produced by the addition operation.
  LIBC_INLINE constexpr WordType add(const BigInt &x) {
    SumCarry<WordType> s{0, 0};
    for (size_t i = 0; i < WORD_COUNT; ++i) {
      s = add_with_carry(val[i], x.val[i], s.carry);
      val[i] = s.sum;
    }
    return s.carry;
  }

  LIBC_INLINE constexpr BigInt operator+(const BigInt &other) const {
    BigInt result;
    SumCarry<WordType> s{0, 0};
    for (size_t i = 0; i < WORD_COUNT; ++i) {
      s = add_with_carry(val[i], other.val[i], s.carry);
      result.val[i] = s.sum;
    }
    return result;
  }

  // This will only apply when initializing a variable from constant values, so
  // it will always use the constexpr version of add_with_carry.
  LIBC_INLINE constexpr BigInt operator+(BigInt &&other) const {
    BigInt result;
    SumCarry<WordType> s{0, 0};
    for (size_t i = 0; i < WORD_COUNT; ++i) {
      s = add_with_carry(val[i], other.val[i], s.carry);
      result.val[i] = s.sum;
    }
    return result;
  }

  LIBC_INLINE constexpr BigInt &operator+=(const BigInt &other) {
    add(other); // Returned carry value is ignored.
    return *this;
  }

  // Subtract x to this number and store the result in this number.
  // Returns the carry value produced by the subtraction operation.
  LIBC_INLINE constexpr WordType sub(const BigInt &x) {
    DiffBorrow<WordType> d{0, 0};
    for (size_t i = 0; i < WORD_COUNT; ++i) {
      d = sub_with_borrow(val[i], x.val[i], d.borrow);
      val[i] = d.diff;
    }
    return d.borrow;
  }

  LIBC_INLINE constexpr BigInt operator-(const BigInt &other) const {
    BigInt result;
    DiffBorrow<WordType> d{0, 0};
    for (size_t i = 0; i < WORD_COUNT; ++i) {
      d = sub_with_borrow(val[i], other.val[i], d.borrow);
      result.val[i] = d.diff;
    }
    return result;
  }

  LIBC_INLINE constexpr BigInt operator-(BigInt &&other) const {
    BigInt result;
    DiffBorrow<WordType> d{0, 0};
    for (size_t i = 0; i < WORD_COUNT; ++i) {
      d = sub_with_borrow(val[i], other.val[i], d.borrow);
      result.val[i] = d.diff;
    }
    return result;
  }

  LIBC_INLINE constexpr BigInt &operator-=(const BigInt &other) {
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
    for (size_t i = 0; i < WORD_COUNT; ++i) {
      NumberPair<WordType> prod = internal::full_mul(val[i], x);
      BigInt<2 * WORD_SIZE, Signed, WordType> tmp({prod.lo, prod.hi});
      const WordType carry = partial_sum.add(tmp);
      val[i] = partial_sum.val[0];
      partial_sum.val[0] = partial_sum.val[1];
      partial_sum.val[1] = carry;
    }
    return partial_sum.val[1];
  }

  LIBC_INLINE constexpr BigInt operator*(const BigInt &other) const {
    if constexpr (Signed) {
      BigInt<Bits, false, WordType> a(*this);
      BigInt<Bits, false, WordType> b(other);
      const bool a_neg = a.is_neg();
      const bool b_neg = b.is_neg();
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
        BigInt result(0);
        BigInt<2 * WORD_SIZE, Signed, WordType> partial_sum(0);
        WordType carry = 0;
        for (size_t i = 0; i < WORD_COUNT; ++i) {
          for (size_t j = 0; j <= i; j++) {
            NumberPair<WordType> prod =
                internal::full_mul(val[j], other.val[i - j]);
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
        NumberPair<WordType> prod =
            internal::full_mul(val[j], other.val[i - j]);
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
  LIBC_INLINE constexpr BigInt quick_mul_hi(const BigInt &other) const {
    BigInt result(0);
    BigInt<2 * WORD_SIZE, Signed, WordType> partial_sum(0);
    WordType carry = 0;
    // First round of accumulation for those at WORD_COUNT - 1 in the full
    // product.
    for (size_t i = 0; i < WORD_COUNT; ++i) {
      NumberPair<WordType> prod =
          internal::full_mul(val[i], other.val[WORD_COUNT - 1 - i]);
      BigInt<2 * WORD_SIZE, Signed, WordType> tmp({prod.lo, prod.hi});
      carry += partial_sum.add(tmp);
    }
    for (size_t i = WORD_COUNT; i < 2 * WORD_COUNT - 1; ++i) {
      partial_sum.val[0] = partial_sum.val[1];
      partial_sum.val[1] = carry;
      carry = 0;
      for (size_t j = i - WORD_COUNT + 1; j < WORD_COUNT; ++j) {
        NumberPair<WordType> prod =
            internal::full_mul(val[j], other.val[i - j]);
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
    BigInt result = 1;
    BigInt cur_power = *this;

    while (power > 0) {
      if ((power % 2) > 0)
        result *= cur_power;
      power >>= 1;
      cur_power *= cur_power;
    }
    *this = result;
  }

  // TODO: Make division work correctly for signed integers.

  // div takes another BigInt of the same size and divides this by it. The value
  // of this will be set to the quotient, and the return value is the remainder.
  LIBC_INLINE constexpr cpp::optional<BigInt> div(const BigInt &other) {
    BigInt remainder(0);
    if (*this < other) {
      remainder = *this;
      *this = BigInt(0);
      return remainder;
    }
    if (other == 1) {
      return remainder;
    }
    if (other == 0) {
      return cpp::nullopt;
    }

    BigInt quotient(0);
    BigInt subtractor = other;
    int cur_bit = static_cast<int>(subtractor.clz() - this->clz());
    subtractor.shift_left(cur_bit);

    for (; cur_bit >= 0 && *this > 0; --cur_bit, subtractor.shift_right(1)) {
      if (*this >= subtractor) {
        this->sub(subtractor);
        quotient = quotient | (BigInt(1) << cur_bit);
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
  LIBC_INLINE constexpr cpp::optional<BigInt>
  div_uint_half_times_pow_2(internal::half_width_t<WordType> x, size_t e) {
    BigInt remainder(0);

    if (x == 0) {
      return cpp::nullopt;
    }
    if (e >= Bits) {
      remainder = *this;
      *this = BigInt<Bits, false, WordType>(0);
      return remainder;
    }

    BigInt quotient(0);
    WordType x_word = static_cast<WordType>(x);
    constexpr size_t LOG2_WORD_SIZE = cpp::bit_width(WORD_SIZE) - 1;
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

  LIBC_INLINE constexpr BigInt operator/(const BigInt &other) const {
    BigInt result(*this);
    result.div(other);
    return result;
  }

  LIBC_INLINE constexpr BigInt &operator/=(const BigInt &other) {
    div(other);
    return *this;
  }

  LIBC_INLINE constexpr BigInt operator%(const BigInt &other) const {
    BigInt result(*this);
    return *result.div(other);
  }

  LIBC_INLINE constexpr BigInt &operator*=(const BigInt &other) {
    *this = *this * other;
    return *this;
  }

  // TODO: remove and use cpp::countl_zero below.
  [[nodiscard]] LIBC_INLINE constexpr int clz() const {
    constexpr int word_digits = cpp::numeric_limits<word_type>::digits;
    int leading_zeroes = 0;
    for (auto i = val.size(); i > 0;) {
      --i;
      const int zeroes = cpp::countl_zero(val[i]);
      leading_zeroes += zeroes;
      if (zeroes != word_digits)
        break;
    }
    return leading_zeroes;
  }

  // TODO: remove and use cpp::countr_zero below.
  [[nodiscard]] LIBC_INLINE constexpr int ctz() const {
    constexpr int word_digits = cpp::numeric_limits<word_type>::digits;
    int trailing_zeroes = 0;
    for (auto word : val) {
      const int zeroes = cpp::countr_zero(word);
      trailing_zeroes += zeroes;
      if (zeroes != word_digits)
        break;
    }
    return trailing_zeroes;
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
#ifdef LIBC_TYPES_HAS_INT128
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
#endif // LIBC_TYPES_HAS_INT128
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

  LIBC_INLINE constexpr BigInt operator<<(size_t s) const {
    BigInt result(*this);
    result.shift_left(s);
    return result;
  }

  LIBC_INLINE constexpr BigInt &operator<<=(size_t s) {
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
#ifdef LIBC_TYPES_HAS_INT128
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
#endif // LIBC_TYPES_HAS_INT128

    if (LIBC_UNLIKELY(s == 0))
      return;
    const size_t drop = s / WORD_SIZE;  // Number of words to drop
    const size_t shift = s % WORD_SIZE; // Bit shift in the remaining words.

    size_t i = 0;
    WordType sign = Signed ? is_neg() : 0;

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

  LIBC_INLINE constexpr BigInt operator>>(size_t s) const {
    BigInt result(*this);
    result.shift_right(s);
    return result;
  }

  LIBC_INLINE constexpr BigInt &operator>>=(size_t s) {
    shift_right(s);
    return *this;
  }

#define DEFINE_BINOP(OP)                                                       \
  LIBC_INLINE friend constexpr BigInt operator OP(const BigInt &lhs,           \
                                                  const BigInt &rhs) {         \
    BigInt result;                                                             \
    for (size_t i = 0; i < WORD_COUNT; ++i)                                    \
      result[i] = lhs[i] OP rhs[i];                                            \
    return result;                                                             \
  }                                                                            \
  LIBC_INLINE friend constexpr BigInt operator OP##=(BigInt &lhs,              \
                                                     const BigInt &rhs) {      \
    for (size_t i = 0; i < WORD_COUNT; ++i)                                    \
      lhs[i] OP## = rhs[i];                                                    \
    return lhs;                                                                \
  }

  DEFINE_BINOP(&)
  DEFINE_BINOP(|)
  DEFINE_BINOP(^)

#undef DEFINE_BINOP

  LIBC_INLINE constexpr BigInt operator~() const {
    BigInt result;
    for (size_t i = 0; i < WORD_COUNT; ++i)
      result[i] = ~val[i];
    return result;
  }

  LIBC_INLINE constexpr BigInt operator-() const {
    BigInt result = ~(*this);
    result.add(BigInt(1));
    return result;
  }

  LIBC_INLINE friend constexpr bool operator==(const BigInt &lhs,
                                               const BigInt &rhs) {
    for (size_t i = 0; i < WORD_COUNT; ++i)
      if (lhs.val[i] != rhs.val[i])
        return false;
    return true;
  }

  LIBC_INLINE friend constexpr bool operator!=(const BigInt &lhs,
                                               const BigInt &rhs) {
    return !(lhs == rhs);
  }

private:
  LIBC_INLINE friend constexpr int cmp(const BigInt &lhs, const BigInt &rhs) {
    const auto compare = [](WordType a, WordType b) {
      return a == b ? 0 : a > b ? 1 : -1;
    };
    if constexpr (Signed) {
      const bool lhs_is_neg = lhs.is_neg();
      const bool rhs_is_neg = rhs.is_neg();
      if (lhs_is_neg != rhs_is_neg)
        return rhs_is_neg ? 1 : -1;
    }
    for (size_t i = WORD_COUNT; i-- > 0;)
      if (auto cmp = compare(lhs[i], rhs[i]); cmp != 0)
        return cmp;
    return 0;
  }

public:
  LIBC_INLINE friend constexpr bool operator>(const BigInt &lhs,
                                              const BigInt &rhs) {
    return cmp(lhs, rhs) > 0;
  }
  LIBC_INLINE friend constexpr bool operator>=(const BigInt &lhs,
                                               const BigInt &rhs) {
    return cmp(lhs, rhs) >= 0;
  }
  LIBC_INLINE friend constexpr bool operator<(const BigInt &lhs,
                                              const BigInt &rhs) {
    return cmp(lhs, rhs) < 0;
  }
  LIBC_INLINE friend constexpr bool operator<=(const BigInt &lhs,
                                               const BigInt &rhs) {
    return cmp(lhs, rhs) <= 0;
  }

  LIBC_INLINE constexpr BigInt &operator++() {
    add(BigInt(1));
    return *this;
  }

  LIBC_INLINE constexpr BigInt operator++(int) {
    BigInt oldval(*this);
    add(BigInt(1));
    return oldval;
  }

  LIBC_INLINE constexpr BigInt &operator--() {
    sub(BigInt(1));
    return *this;
  }

  LIBC_INLINE constexpr BigInt operator--(int) {
    BigInt oldval(*this);
    sub(BigInt(1));
    return oldval;
  }

  // Return the i-th word of the number.
  LIBC_INLINE constexpr const WordType &operator[](size_t i) const {
    return val[i];
  }

  // Return the i-th word of the number.
  LIBC_INLINE constexpr WordType &operator[](size_t i) { return val[i]; }

  LIBC_INLINE WordType *data() { return val; }

  LIBC_INLINE const WordType *data() const { return val; }
};

namespace internal {
// We default BigInt's WordType to 'uint64_t' or 'uint32_t' depending on type
// availability.
template <size_t Bits>
struct WordTypeSelector : cpp::type_identity<
#ifdef LIBC_TYPES_HAS_INT64
                              uint64_t
#else
                              uint32_t
#endif // LIBC_TYPES_HAS_INT64
                              > {
};
// Except if we request 32 bits explicitly.
template <> struct WordTypeSelector<32> : cpp::type_identity<uint32_t> {};
template <size_t Bits>
using WordTypeSelectorT = typename WordTypeSelector<Bits>::type;
} // namespace internal

template <size_t Bits>
using UInt = BigInt<Bits, false, internal::WordTypeSelectorT<Bits>>;

template <size_t Bits>
using Int = BigInt<Bits, true, internal::WordTypeSelectorT<Bits>>;

// Provides limits of U/Int<128>.
template <> class cpp::numeric_limits<UInt<128>> {
public:
  LIBC_INLINE static constexpr UInt<128> max() {
    return UInt<128>({0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff});
  }
  LIBC_INLINE static constexpr UInt<128> min() { return UInt<128>(0); }
  // Meant to match std::numeric_limits interface.
  // NOLINTNEXTLINE(readability-identifier-naming)
  LIBC_INLINE_VAR static constexpr int digits = 128;
};

template <> class cpp::numeric_limits<Int<128>> {
public:
  LIBC_INLINE static constexpr Int<128> max() {
    return Int<128>({0xffff'ffff'ffff'ffff, 0x7fff'ffff'ffff'ffff});
  }
  LIBC_INLINE static constexpr Int<128> min() {
    return Int<128>({0, 0x8000'0000'0000'0000});
  }
  // Meant to match std::numeric_limits interface.
  // NOLINTNEXTLINE(readability-identifier-naming)
  LIBC_INLINE_VAR static constexpr int digits = 128;
};

// type traits to determine whether a T is a BigInt.
template <typename T> struct is_big_int : cpp::false_type {};

template <size_t Bits, bool Signed, typename T>
struct is_big_int<BigInt<Bits, Signed, T>> : cpp::true_type {};

template <class T>
LIBC_INLINE_VAR constexpr bool is_big_int_v = is_big_int<T>::value;

namespace cpp {

// Specialization of cpp::bit_cast ('bit.h') from T to BigInt.
template <typename To, typename From>
LIBC_INLINE constexpr cpp::enable_if_t<
    (sizeof(To) == sizeof(From)) && cpp::is_trivially_copyable<To>::value &&
        cpp::is_trivially_copyable<From>::value && is_big_int<To>::value,
    To>
bit_cast(const From &from) {
  To out;
  using Storage = decltype(out.val);
  out.val = cpp::bit_cast<Storage>(from);
  return out;
}

// Specialization of cpp::bit_cast ('bit.h') from BigInt to T.
template <typename To, size_t Bits>
LIBC_INLINE constexpr cpp::enable_if_t<
    sizeof(To) == sizeof(UInt<Bits>) &&
        cpp::is_trivially_constructible<To>::value &&
        cpp::is_trivially_copyable<To>::value &&
        cpp::is_trivially_copyable<UInt<Bits>>::value,
    To>
bit_cast(const UInt<Bits> &from) {
  return cpp::bit_cast<To>(from.val);
}

// Specialization of cpp::has_single_bit ('bit.h') for BigInt.
template <typename T>
[[nodiscard]] LIBC_INLINE constexpr cpp::enable_if_t<is_big_int_v<T>, bool>
has_single_bit(T value) {
  int bits = 0;
  for (auto word : value.val) {
    if (word == 0)
      continue;
    bits += popcount(word);
    if (bits > 1)
      return false;
  }
  return bits == 1;
}

// Specialization of cpp::countr_zero ('bit.h') for BigInt.
template <typename T>
[[nodiscard]] LIBC_INLINE constexpr cpp::enable_if_t<is_big_int_v<T>, int>
countr_zero(const T &value) {
  return value.ctz();
}

// Specialization of cpp::countl_zero ('bit.h') for BigInt.
template <typename T>
[[nodiscard]] LIBC_INLINE constexpr cpp::enable_if_t<is_big_int_v<T>, int>
countl_zero(const T &value) {
  return value.clz();
}

// Specialization of cpp::countl_one ('bit.h') for BigInt.
template <typename T>
[[nodiscard]] LIBC_INLINE constexpr cpp::enable_if_t<is_big_int_v<T>, int>
countl_one(T value) {
  // TODO : Implement a faster version not involving operator~.
  return cpp::countl_zero<T>(~value);
}

// Specialization of cpp::countr_one ('bit.h') for BigInt.
template <typename T>
[[nodiscard]] LIBC_INLINE constexpr cpp::enable_if_t<is_big_int_v<T>, int>
countr_one(T value) {
  // TODO : Implement a faster version not involving operator~.
  return cpp::countr_zero<T>(~value);
}

// Specialization of cpp::bit_width ('bit.h') for BigInt.
template <typename T>
[[nodiscard]] LIBC_INLINE constexpr cpp::enable_if_t<is_big_int_v<T>, int>
bit_width(T value) {
  return cpp::numeric_limits<T>::digits - cpp::countl_zero(value);
}

// Forward-declare rotr so that rotl can use it.
template <typename T>
[[nodiscard]] LIBC_INLINE constexpr cpp::enable_if_t<is_big_int_v<T>, T>
rotr(T value, int rotate);

// Specialization of cpp::rotl ('bit.h') for BigInt.
template <typename T>
[[nodiscard]] LIBC_INLINE constexpr cpp::enable_if_t<is_big_int_v<T>, T>
rotl(T value, int rotate) {
  constexpr unsigned N = cpp::numeric_limits<T>::digits;
  rotate = rotate % N;
  if (!rotate)
    return value;
  if (rotate < 0)
    return cpp::rotr<T>(value, -rotate);
  return (value << rotate) | (value >> (N - rotate));
}

// Specialization of cpp::rotr ('bit.h') for BigInt.
template <typename T>
[[nodiscard]] LIBC_INLINE constexpr cpp::enable_if_t<is_big_int_v<T>, T>
rotr(T value, int rotate) {
  constexpr unsigned N = cpp::numeric_limits<T>::digits;
  rotate = rotate % N;
  if (!rotate)
    return value;
  if (rotate < 0)
    return cpp::rotl<T>(value, -rotate);
  return (value >> rotate) | (value << (N - rotate));
}

} // namespace cpp

// Specialization of mask_trailing_ones ('math_extras.h') for BigInt.
template <typename T, size_t count>
LIBC_INLINE constexpr cpp::enable_if_t<is_big_int_v<T>, T>
mask_trailing_ones() {
  static_assert(!T::SIGNED);
  if (count == 0)
    return T();
  constexpr unsigned T_BITS = CHAR_BIT * sizeof(T);
  static_assert(count <= T_BITS && "Invalid bit index");
  using word_type = typename T::word_type;
  T out;
  constexpr int CHUNK_INDEX_CONTAINING_BIT =
      static_cast<int>(count / T::WORD_SIZE);
  int index = 0;
  for (auto &word : out.val) {
    if (index < CHUNK_INDEX_CONTAINING_BIT)
      word = -1;
    else if (index > CHUNK_INDEX_CONTAINING_BIT)
      word = 0;
    else
      word = mask_trailing_ones<word_type, count % T::WORD_SIZE>();
    ++index;
  }
  return out;
}

// Specialization of mask_leading_ones ('math_extras.h') for BigInt.
template <typename T, size_t count>
LIBC_INLINE constexpr cpp::enable_if_t<is_big_int_v<T>, T> mask_leading_ones() {
  static_assert(!T::SIGNED);
  if (count == 0)
    return T();
  constexpr unsigned T_BITS = CHAR_BIT * sizeof(T);
  static_assert(count <= T_BITS && "Invalid bit index");
  using word_type = typename T::word_type;
  T out;
  constexpr int CHUNK_INDEX_CONTAINING_BIT =
      static_cast<int>((T::BITS - count - 1ULL) / T::WORD_SIZE);
  int index = 0;
  for (auto &word : out.val) {
    if (index < CHUNK_INDEX_CONTAINING_BIT)
      word = 0;
    else if (index > CHUNK_INDEX_CONTAINING_BIT)
      word = -1;
    else
      word = mask_leading_ones<word_type, count % T::WORD_SIZE>();
    ++index;
  }
  return out;
}

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC___SUPPORT_UINT_H
