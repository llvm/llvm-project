//===-- Utilities to convert floating point values to string ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_FLOAT_TO_STRING_H
#define LLVM_LIBC_SRC_SUPPORT_FLOAT_TO_STRING_H

#include <stdint.h>

#include "src/__support/CPP/type_traits.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/UInt.h"
#include "src/__support/ryu_constants.h"

// This implementation is based on the Ryu Printf algorithm by Ulf Adams:
// Ulf Adams. 2019. Ryū revisited: printf floating point conversion.
// Proc. ACM Program. Lang. 3, OOPSLA, Article 169 (October 2019), 23 pages.
// https://doi.org/10.1145/3360595

// This version is modified to require significantly less memory (it doesn't use
// a large buffer to store the result).

// The general concept of this algorithm is as follows:
// We want to calculate a 9 digit segment of a floating point number using this
// formula: floor((mantissa * 2^exponent)/10^i) % 10^9.
// To do so normally would involve large integers (~1000 bits for doubles), so
// we use a shortcut. We can avoid calculating 2^exponent / 10^i by using a
// lookup table. The resulting intermediate value needs to be about 192 bits to
// store the result with enough precision. Since this is all being done with
// integers for appropriate precision, we would run into a problem if
// i > exponent since then 2^exponent / 10^i would be less than 1. To correct
// for this, the actual calculation done is 2^(exponent + c) / 10^i, and then
// when multiplying by the mantissa we reverse this by dividing by 2^c, like so:
// floor((mantissa * table[exponent][i])/(2^c)) % 10^9.
// This gives a 9 digit value, which is small enough to fit in a 32 bit integer,
// and that integer is converted into a string as normal, and called a block. In
// this implementation, the most recent block is buffered, so that if rounding
// is necessary the block can be adjusted before being written to the output.
// Any block that is all 9s adds one to the max block counter and doesn't clear
// the buffer because they can cause the block above them to be rounded up.

namespace __llvm_libc {

using BlockInt = uint32_t;
constexpr size_t BLOCK_SIZE = 9;

using MantissaInt = fputil::FPBits<long double>::UIntType;

constexpr size_t POW10_ADDITIONAL_BITS_CALC = 128;
constexpr size_t POW10_ADDITIONAL_BITS_TABLE = 120;

constexpr size_t MID_INT_SIZE = 192;

namespace internal {

// Returns floor(log_10(2^e)); requires 0 <= e <= 1650.
constexpr inline uint32_t log10_pow2(const uint32_t e) {
  // The first value this approximation fails for is 2^1651 which is just
  // greater than 10^297. assert(e >= 0); assert(e <= 1650);
  return (e * 78913) >> 18;
}

// Returns 1 + floor(log_10(2^e). This could technically be off by 1 if any
// power of 2 was also a power of 10, but since that doesn't exist this is
// always accurate. This is used to calculate the maximum number of base-10
// digits a given e-bit number could have.
constexpr inline uint32_t ceil_log10_pow2(const uint32_t e) {
  return log10_pow2(e) + 1;
}

// Returns the maximum number of 9 digit blocks a number described by the given
// index (which is ceil(exponent/16)) and mantissa width could need.
constexpr inline uint32_t length_for_num(const uint32_t idx,
                                         const uint32_t mantissa_width) {
  //+8 to round up when dividing by 9
  return (ceil_log10_pow2(16 * idx) + ceil_log10_pow2(mantissa_width) +
          (BLOCK_SIZE - 1)) /
         BLOCK_SIZE;
  // return (ceil_log10_pow2(16 * idx + mantissa_width) + 8) / 9;
}

// The formula for the table when i is positive (or zero) is as follows:
// floor(10^(-9i) * 2^(e + c_1) + 1) % (10^9 * 2^c_1)
// Rewritten slightly we get:
// floor(5^(-9i) * 2^(e + c_1 - 9i) + 1) % (10^9 * 2^c_1)

template <size_t INT_SIZE>
constexpr inline cpp::UInt<MID_INT_SIZE>
get_table_positive(int exponent, size_t i, const size_t constant) {
  // INT_SIZE is the size of int that is used for the internal calculations of
  // this function. It should be large enough to hold 2^(exponent+constant), so
  // ~1000 for double and ~16000 for long double. Be warned that the time
  // complexity of exponentiation is O(n^2 * log_2(m)) where n is the number of
  // bits in the number being exponentiated and m is the exponent.
  int shift_amount = exponent + constant - (9 * i);
  if (shift_amount < 0) {
    return 1;
  }
  cpp::UInt<INT_SIZE> num(0);
  // MOD_SIZE is one of the limiting factors for how big the constant argument
  // can get, since it needs to be small enough to fit in the result UInt,
  // otherwise we'll get truncation on return.
  const cpp::UInt<INT_SIZE> MOD_SIZE =
      (cpp::UInt<INT_SIZE>(1) << constant) * 1000000000;
  constexpr uint64_t FIVE_EXP_NINE = 1953125;

  num = cpp::UInt<INT_SIZE>(1) << (shift_amount);
  if (i > 0) {
    cpp::UInt<INT_SIZE> fives(FIVE_EXP_NINE);
    fives.pow_n(i);
    num = num / fives;
  }

  num = num + 1;
  if (num > MOD_SIZE) {
    num = num % MOD_SIZE;
  }
  return num;
}

// The formula for the table when i is negative (or zero) is as follows:
// floor(10^(-9i) * 2^(c_0 - e)) % (10^9 * 2^c_0)
// Since we know i is always negative, we just take it as unsigned and treat it
// as negative. We do the same with exponent, while they're both always negative
// in theory, in practice they're converted to positive for simpler
// calculations.
// The formula being used looks more like this:
// floor(10^(9*(-i)) * 2^(c_0 + (-e))) % (10^9 * 2^c_0)
constexpr inline cpp::UInt<MID_INT_SIZE>
get_table_negative(int exponent, size_t i, const size_t constant) {
  constexpr size_t INT_SIZE = 1024;
  int shift_amount = constant - exponent;
  cpp::UInt<INT_SIZE> num(1);
  // const cpp::UInt<INT_SIZE> MOD_SIZE =
  //     (cpp::UInt<INT_SIZE>(1) << constant) * 1000000000;

  constexpr uint64_t TEN_EXP_NINE = 1000000000;
  constexpr uint64_t FIVE_EXP_NINE = 1953125;
  size_t ten_blocks = i;
  size_t five_blocks = 0;
  if (shift_amount < 0) {
    int block_shifts = (-shift_amount) / 9;
    if (block_shifts < static_cast<int>(ten_blocks)) {
      ten_blocks = ten_blocks - block_shifts;
      five_blocks = block_shifts;
      shift_amount = shift_amount + (block_shifts * 9);
    } else {
      ten_blocks = 0;
      five_blocks = i;
      shift_amount = shift_amount + (i * 9);
    }
  }

  if (five_blocks > 0) {
    cpp::UInt<INT_SIZE> fives(FIVE_EXP_NINE);
    fives.pow_n(five_blocks);
    num *= fives;
  }
  if (ten_blocks > 0) {
    cpp::UInt<INT_SIZE> tens(TEN_EXP_NINE);
    tens.pow_n(ten_blocks);
    num *= tens;
  }

  if (shift_amount > 0) {
    num = num << shift_amount;
  } else {
    num = num >> (-shift_amount);
  }
  // if (num > MOD_SIZE) {
  //   num = num % MOD_SIZE;
  // }
  return num;
}

static inline uint32_t fast_uint_mod_1e9(const cpp::UInt<MID_INT_SIZE> &val) {
  // The formula for mult_const is:
  //  1 + floor((2^(bits in target integer size + log_2(divider))) / divider)
  // Where divider is 10^9 and target integer size is 128.
  const cpp::UInt<MID_INT_SIZE> mult_const(
      {0x31680A88F8953031u, 0x89705F4136B4A597u, 0});
  const auto middle = (mult_const * val);
  const uint64_t result = static_cast<uint64_t>(middle[2]);
  const uint32_t shifted = result >> 29;
  return static_cast<uint32_t>(val) - (1000000000 * shifted);
}

static inline uint32_t mul_shift_mod_1e9(const MantissaInt mantissa,
                                         const cpp::UInt<MID_INT_SIZE> &large,
                                         const int32_t shift_amount) {
  constexpr size_t MANT_INT_SIZE = sizeof(MantissaInt) * 8;
  cpp::UInt<MID_INT_SIZE + MANT_INT_SIZE> val(large);
  // TODO: Find a better way to force __uint128_t to be UInt<128>
  cpp::UInt<MANT_INT_SIZE> wide_mant(0);
  wide_mant[0] = mantissa & (uint64_t(-1));
  wide_mant[1] = mantissa >> 64;
  val = (val * wide_mant) >> shift_amount;
  return fast_uint_mod_1e9(val);
}

} // namespace internal

// Convert floating point values to their string representation.
// Because the result may not fit in a reasonably sized array, the caller must
// request blocks of digits and convert them from integers to strings themself.
// Blocks contain the most digits that can be stored in an BlockInt. This is 9
// digits for a 32 bit int and 18 digits for a 64 bit int.
// The intended use pattern is to create a FloatToString object of the
// appropriate type, then call get_positive_blocks to get an approximate number
// of blocks there are before the decimal point. Now the client code can start
// calling get_positive_block in a loop from the number of positive blocks to
// zero. This will give all digits before the decimal point. Then the user can
// start calling get_negative_block in a loop from 0 until the number of digits
// they need is reached. As an optimization, the client can use
// zero_blocks_after_point to find the number of blocks that are guaranteed to
// be zero after the decimal point and before the non-zero digits. Additionally,
// is_lowest_block will return if the current block is the lowest non-zero
// block.
template <typename T, cpp::enable_if_t<cpp::is_floating_point_v<T>, int> = 0>
class FloatToString {
  fputil::FPBits<T> float_bits;
  bool is_negative;
  int exponent;
  MantissaInt mantissa;

  static constexpr int MANT_WIDTH = fputil::MantissaWidth<T>::VALUE;
  static constexpr int EXP_BIAS = fputil::FPBits<T>::EXPONENT_BIAS;

  // constexpr void init_convert();

public:
  constexpr FloatToString<T>(T init_float) : float_bits(init_float) {
    is_negative = float_bits.get_sign();
    exponent = float_bits.get_exponent();
    mantissa = float_bits.get_explicit_mantissa();

    // Handle the exponent for numbers with a 0 exponent.
    if (exponent == -EXP_BIAS) {
      if (mantissa > 0) { // Subnormals
        ++exponent;
      } else { // Zeroes
        exponent = 0;
      }
    }

    // Adjust for the width of the mantissa.
    exponent -= MANT_WIDTH;

    // init_convert();
  }

  constexpr bool is_nan() { return float_bits.is_nan(); }
  constexpr bool is_inf() { return float_bits.is_inf(); }
  constexpr bool is_inf_or_nan() { return float_bits.is_inf_or_nan(); }

  // get_block returns an integer that represents the digits in the requested
  // block.
  constexpr BlockInt get_positive_block(int block_index) {
    if (exponent >= -MANT_WIDTH) {
      // idx is ceil(exponent/16) or 0 if exponent is negative. This is used to
      // find the coarse section of the POW10_SPLIT table that will be used to
      // calculate the 9 digit window, as well as some other related values.
      const uint32_t idx =
          exponent < 0 ? 0 : static_cast<uint32_t>(exponent + 15) / 16;

      uint32_t temp_shift_amount =
          POW10_ADDITIONAL_BITS_TABLE + (16 * idx) - exponent;
      const uint32_t shift_amount = temp_shift_amount;
      // shift_amount = -(c0 - exponent) = c_0 + 16 * ceil(exponent/16) -
      // exponent

      int32_t i = block_index;
      cpp::UInt<MID_INT_SIZE> val;
      val = POW10_SPLIT[POW10_OFFSET[idx] + i];

      const uint32_t digits =
          internal::mul_shift_mod_1e9(mantissa, val, (int32_t)(shift_amount));
      return digits;
    } else {
      return 0;
    }
  }
  constexpr BlockInt get_negative_block(int block_index) {
    if (exponent < 0) {
      const int32_t idx = -exponent / 16;
      uint32_t i = block_index;
      // if the requested block is zero
      if (i < MIN_BLOCK_2[idx]) {
        return 0;
      }
      const int32_t shift_amount =
          POW10_ADDITIONAL_BITS_TABLE + (-exponent - 16 * idx);
      const uint32_t p = POW10_OFFSET_2[idx] + i - MIN_BLOCK_2[idx];
      // If every digit after the requested block is zero.
      if (p >= POW10_OFFSET_2[idx + 1]) {
        return 0;
      }

      cpp::UInt<MID_INT_SIZE> table_val = POW10_SPLIT_2[p];
      // cpp::UInt<MID_INT_SIZE> calculated_val =
      //     get_table_negative(idx * 16, i + 1, POW10_ADDITIONAL_BITS_CALC);
      uint32_t digits =
          internal::mul_shift_mod_1e9(mantissa, table_val, shift_amount);
      return digits;
    } else {
      return 0;
    }
  }
  constexpr size_t get_positive_blocks() {
    if (exponent >= -MANT_WIDTH) {
      const uint32_t idx =
          exponent < 0 ? 0 : static_cast<uint32_t>(exponent + 15) / 16;
      const uint32_t len = internal::length_for_num(idx, MANT_WIDTH);
      return len;
    } else {
      return 0;
    }
  }

  // This takes the index of a block after the decimal point (a negative block)
  // and return if it's sure that all of the digits after it are zero.
  constexpr bool is_lowest_block(size_t block_index) {
    const int32_t idx = -exponent / 16;
    const uint32_t p = POW10_OFFSET_2[idx] + block_index - MIN_BLOCK_2[idx];
    // If the remaining digits are all 0, then this is the lowest block.
    return p >= POW10_OFFSET_2[idx + 1];
  }

  constexpr size_t zero_blocks_after_point() {
    return MIN_BLOCK_2[-exponent / 16];
  }
};

// template <> constexpr void FloatToString<float>::init_convert() { ; }

// template <> constexpr void FloatToString<double>::init_convert() { ; }

// template <> constexpr void FloatToString<long double>::init_convert() {
//   // TODO: More here.
//   ;
// }

template <>
constexpr size_t FloatToString<long double>::zero_blocks_after_point() {
  return 0;
}

template <>
constexpr bool FloatToString<long double>::is_lowest_block(size_t block_index) {
  return block_index < 0;
}

template <>
constexpr BlockInt
FloatToString<long double>::get_positive_block(int block_index) {
  if (exponent >= -MANT_WIDTH) {
    const uint32_t pos_exp = (exponent < 0 ? 0 : exponent);

    uint32_t temp_shift_amount =
        POW10_ADDITIONAL_BITS_CALC + pos_exp - exponent;
    const uint32_t shift_amount = temp_shift_amount;
    // shift_amount = -(c0 - exponent) = c_0 + 16 * ceil(exponent/16) -
    // exponent

    int32_t i = block_index;
    cpp::UInt<MID_INT_SIZE> val;
    if (exponent + POW10_ADDITIONAL_BITS_CALC < 1024) {
      val = internal::get_table_positive<1024>(pos_exp, i,
                                               POW10_ADDITIONAL_BITS_CALC);
    } else if (exponent + POW10_ADDITIONAL_BITS_CALC < 4096) {
      val = internal::get_table_positive<4096>(pos_exp, i,
                                               POW10_ADDITIONAL_BITS_CALC);
    } else if (exponent + POW10_ADDITIONAL_BITS_CALC < 8192) {
      val = internal::get_table_positive<8192>(pos_exp, i,
                                               POW10_ADDITIONAL_BITS_CALC);
    } else {
      val = internal::get_table_positive<16384>(pos_exp, i,
                                                POW10_ADDITIONAL_BITS_CALC);
    }

    const BlockInt digits =
        internal::mul_shift_mod_1e9(mantissa, val, (int32_t)(shift_amount));
    return digits;
  } else {
    return 0;
  }
}

template <>
constexpr BlockInt
FloatToString<long double>::get_negative_block(int block_index) {
  if (exponent < 0) {
    const int32_t idx = -exponent / 16;
    uint32_t i = -1 - block_index;
    // if the requested block is zero
    if (i <= MIN_BLOCK_2[idx]) {
      return 0;
    }
    const int32_t shift_amount =
        POW10_ADDITIONAL_BITS_CALC + (-exponent - 16 * idx);
    const uint32_t p = POW10_OFFSET_2[idx] + i - MIN_BLOCK_2[idx];
    // If every digit after the requested block is zero.
    if (p >= POW10_OFFSET_2[idx + 1]) {
      return 0;
    }

    // cpp::UInt<MID_INT_SIZE> table_val = POW10_SPLIT_2[p];
    cpp::UInt<MID_INT_SIZE> calculated_val = internal::get_table_negative(
        idx * 16, i + 1, POW10_ADDITIONAL_BITS_CALC);
    BlockInt digits =
        internal::mul_shift_mod_1e9(mantissa, calculated_val, shift_amount);
    return digits;
  } else {
    return 0;
  }
}

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SUPPORT_FLOAT_TO_STRING_H
