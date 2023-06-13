//===-- Decimal Float Converter for printf ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_PRINTF_CORE_FLOAT_DEC_CONVERTER_H
#define LLVM_LIBC_SRC_STDIO_PRINTF_CORE_FLOAT_DEC_CONVERTER_H

#include "src/__support/CPP/string_view.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/FloatProperties.h"
#include "src/__support/FPUtil/rounding_mode.h"
#include "src/__support/UInt.h"
#include "src/__support/UInt128.h"
#include "src/__support/common.h"
#include "src/__support/float_to_string.h"
#include "src/__support/integer_to_string.h"
#include "src/__support/libc_assert.h"
#include "src/stdio/printf_core/converter_utils.h"
#include "src/stdio/printf_core/core_structs.h"
#include "src/stdio/printf_core/float_inf_nan_converter.h"
#include "src/stdio/printf_core/writer.h"

#include <inttypes.h>
#include <stddef.h>

namespace __llvm_libc {
namespace printf_core {

using MantissaInt = fputil::FPBits<long double>::UIntType;

// Returns true if value is divisible by 2^p.
template <typename T>
LIBC_INLINE constexpr cpp::enable_if_t<cpp::is_integral_v<T>, bool>
multiple_of_power_of_2(T value, uint32_t p) {
  return (value & ((T(1) << p) - 1)) == 0;
}

constexpr size_t BLOCK_SIZE = 9;
constexpr uint32_t MAX_BLOCK = 999999999;

// constexpr size_t BLOCK_SIZE = 18;
// constexpr uint32_t MAX_BLOCK = 999999999999999999;
constexpr char DECIMAL_POINT = '.';

// This is used to represent which direction the number should be rounded.
enum class RoundDirection { Up, Down, Even };

class PaddingWriter {
  bool left_justified = false;
  bool leading_zeroes = false;
  char sign_char = 0;
  size_t min_width = 0;

public:
  PaddingWriter() {}
  PaddingWriter(const FormatSection &to_conv, char init_sign_char)
      : left_justified((to_conv.flags & FormatFlags::LEFT_JUSTIFIED) > 0),
        leading_zeroes((to_conv.flags & FormatFlags::LEADING_ZEROES) > 0),
        sign_char(init_sign_char),
        min_width(to_conv.min_width > 0 ? to_conv.min_width : 0) {}

  int write_left_padding(Writer *writer, size_t total_digits) {
    // The pattern is (spaces) (sign) (zeroes), but only one of spaces and
    // zeroes can be written, and only if the padding amount is positive.
    int padding_amount = min_width - total_digits - (sign_char > 0 ? 1 : 0);
    if (left_justified || padding_amount < 0) {
      if (sign_char > 0) {
        RET_IF_RESULT_NEGATIVE(writer->write(sign_char));
      }
      return 0;
    }
    if (!leading_zeroes) {
      RET_IF_RESULT_NEGATIVE(writer->write(' ', padding_amount));
    }
    if (sign_char > 0) {
      RET_IF_RESULT_NEGATIVE(writer->write(sign_char));
    }
    if (leading_zeroes) {
      RET_IF_RESULT_NEGATIVE(writer->write('0', padding_amount));
    }
    return 0;
  }

  int write_right_padding(Writer *writer, size_t total_digits) {
    // If and only if the conversion is left justified, there may be trailing
    // spaces.
    int padding_amount = min_width - total_digits - (sign_char > 0 ? 1 : 0);
    if (left_justified && padding_amount > 0) {
      RET_IF_RESULT_NEGATIVE(writer->write(' ', padding_amount));
    }
    return 0;
  }
};

/*
  We only need to round a given segment if all of the segments below it are
  the max (or this is the last segment). This means that we don't have to
  write those initially, we can just keep the most recent non-maximal
  segment and a counter of the number of maximal segments. When we reach a
  non-maximal segment, we write the stored segment as well as as many 9s as
  are necessary. Alternately, if we reach the end and have to round up, then
  we round the stored segment, and write zeroes following it. If this
  crosses the decimal point, then we have to shift it one space to the
  right.
  This FloatWriter class does the buffering and counting, and writes to the
  output when necessary.
*/
class FloatWriter {
  char block_buffer[BLOCK_SIZE]; // The buffer that holds a block.
  size_t buffered_digits = 0;    // The number of digits held in the buffer.
  bool has_written = false;      // True once any digits have been output.
  size_t max_block_count = 0; // The # of blocks of all 9s currently buffered.
  size_t total_digits = 0;    // The number of digits that will be output.
  size_t digits_before_decimal = 0; // The # of digits to write before the '.'
  size_t total_digits_written = 0;  // The # of digits that have been output.
  bool has_decimal_point;           // True if the number has a decimal point.
  Writer *writer;                   // Writes to the final output.
  PaddingWriter padding_writer; // Handles prefixes/padding, uses total_digits.

  int flush_buffer() {
    // Write the most recent buffered block, and mark has_written
    if (!has_written) {
      has_written = true;
      RET_IF_RESULT_NEGATIVE(
          padding_writer.write_left_padding(writer, total_digits));
    }

    // if the decimal point is the next character, or is in the range covered
    // by the buffered block, write the appropriate digits and the decimal
    // point.
    if (total_digits_written < digits_before_decimal &&
        total_digits_written + buffered_digits >= digits_before_decimal &&
        has_decimal_point) {
      size_t digits_to_write = digits_before_decimal - total_digits_written;
      if (digits_to_write > 0) {
        // Write the digits before the decimal point.
        RET_IF_RESULT_NEGATIVE(writer->write({block_buffer, digits_to_write}));
      }
      RET_IF_RESULT_NEGATIVE(writer->write(DECIMAL_POINT));
      if (buffered_digits - digits_to_write > 0) {
        // Write the digits after the decimal point.
        RET_IF_RESULT_NEGATIVE(
            writer->write({block_buffer + digits_to_write,
                           (buffered_digits - digits_to_write)}));
      }
      // add 1 for the decimal point
      total_digits_written += buffered_digits + 1;
      // Mark the buffer as empty.
      buffered_digits = 0;
    }

    // Clear the buffered digits.
    if (buffered_digits > 0) {
      RET_IF_RESULT_NEGATIVE(writer->write({block_buffer, buffered_digits}));
      total_digits_written += buffered_digits;
      buffered_digits = 0;
    }

    // if the decimal point is the next character, or is in the range covered
    // by the max blocks, write the appropriate digits and the decimal point.
    if (total_digits_written < digits_before_decimal &&
        total_digits_written + BLOCK_SIZE * max_block_count >=
            digits_before_decimal &&
        has_decimal_point) {
      size_t digits_to_write = digits_before_decimal - total_digits_written;
      if (digits_to_write > 0) {
        RET_IF_RESULT_NEGATIVE(writer->write('9', digits_to_write));
      }
      RET_IF_RESULT_NEGATIVE(writer->write(DECIMAL_POINT));
      if ((BLOCK_SIZE * max_block_count) - digits_to_write > 0) {
        RET_IF_RESULT_NEGATIVE(writer->write(
            '9', (BLOCK_SIZE * max_block_count) - digits_to_write));
      }
      // add 1 for the decimal point
      total_digits_written += BLOCK_SIZE * max_block_count + 1;
      // clear the buffer of max blocks
      max_block_count = 0;
    }

    // Clear the buffer of max blocks
    if (max_block_count > 0) {
      RET_IF_RESULT_NEGATIVE(writer->write('9', max_block_count * BLOCK_SIZE));
      total_digits_written += max_block_count * BLOCK_SIZE;
      max_block_count = 0;
    }
    return 0;
  }

  cpp::string_view exp_str(int exponent, cpp::span<char> exp_buffer) {

    // -exponent will never overflow because all long double types we support
    // have at most 15 bits of mantissa and the C standard defines an int as
    // being at least 16 bits.
    static_assert(fputil::FloatProperties<long double>::EXPONENT_WIDTH <
                  (sizeof(int) * 8));

    int positive_exponent = exponent < 0 ? -exponent : exponent;
    char exp_sign = exponent < 0 ? '-' : '+';
    auto const int_to_str =
        *IntegerToString::dec(positive_exponent, exp_buffer);

    // IntegerToString writes the digits from right to left so there will be
    // space to the left of int_to_str.
    size_t digits_in_exp = int_to_str.size();
    size_t index = exp_buffer.size() - digits_in_exp - 1;

    // Ensure that at least two digits were written. IntegerToString always
    // writes at least 1 digit (it writes "0" when the input number is 0).
    if (digits_in_exp < 2) {
      exp_buffer[index] = '0';
      --index;
    }

    // Since the exp_buffer has to be sized to handle an intmax_t, it has space
    // for a sign. In this case we're handling the sign on our own since we also
    // want plus signs for positive numbers.
    exp_buffer[index] = exp_sign;

    return cpp::string_view(exp_buffer.data() + index,
                            exp_buffer.size() - index);
  }

public:
  FloatWriter(Writer *init_writer, bool init_has_decimal_point,
              const PaddingWriter &init_padding_writer)
      : has_decimal_point(init_has_decimal_point), writer(init_writer),
        padding_writer(init_padding_writer) {}

  void init(size_t init_total_digits, size_t init_digits_before_decimal) {
    total_digits = init_total_digits;
    digits_before_decimal = init_digits_before_decimal;
  }

  void write_first_block(BlockInt block, bool exp_format = false) {
    char buf[IntegerToString::dec_bufsize<intmax_t>()];
    auto const int_to_str = *IntegerToString::dec(block, buf);
    size_t digits_buffered = int_to_str.size();
    // Block Buffer is guaranteed to not overflow since block cannot have more
    // than BLOCK_SIZE digits.
    // TODO: Replace with memcpy
    for (size_t count = 0; count < digits_buffered; ++count) {
      block_buffer[count] = int_to_str[count];
    }
    buffered_digits = digits_buffered;

    // In the exponent format (%e) we know how many digits will be written even
    // before calculating any blocks, whereas the decimal format (%f) has to
    // write all of the blocks that would come before the decimal place.
    if (!exp_format) {
      total_digits += digits_buffered;
      digits_before_decimal += digits_buffered;
    }
  }

  int write_middle_block(BlockInt block) {
    if (block == MAX_BLOCK) { // Buffer max blocks in case of rounding
      ++max_block_count;
    } else { // If a non-max block has been found
      RET_IF_RESULT_NEGATIVE(flush_buffer());

      // Now buffer the current block. We add 1 + MAX_BLOCK to force the
      // leading zeroes, and drop the leading one. This is probably inefficient,
      // but it works. See https://xkcd.com/2021/
      char buf[IntegerToString::dec_bufsize<intmax_t>()];
      auto const int_to_str =
          *IntegerToString::dec(block + (MAX_BLOCK + 1), buf);
      // TODO: Replace with memcpy
      for (size_t count = 0; count < BLOCK_SIZE; ++count) {
        block_buffer[count] = int_to_str[count + 1];
      }

      buffered_digits = BLOCK_SIZE;
    }
    return 0;
  }

  int write_last_block_dec(BlockInt block, size_t block_digits,
                           RoundDirection round) {
    char end_buff[BLOCK_SIZE];

    char buf[IntegerToString::dec_bufsize<intmax_t>()];
    auto const int_to_str = *IntegerToString::dec(block + (MAX_BLOCK + 1), buf);

    // copy the last block_digits characters into the start of end_buff.
    // TODO: Replace with memcpy
    for (int count = block_digits - 1; count >= 0; --count) {
      end_buff[count] = int_to_str[count + 1 + (BLOCK_SIZE - block_digits)];
    }

    char low_digit;
    if (block_digits > 0) {
      low_digit = end_buff[block_digits - 1];
    } else if (max_block_count > 0) {
      low_digit = '9';
    } else {
      low_digit = block_buffer[buffered_digits - 1];
    }

    // Round up
    if (round == RoundDirection::Up ||
        (round == RoundDirection::Even && low_digit % 2 != 0)) {
      bool has_carry = true;
      // handle the low block that we're adding
      for (int count = block_digits - 1; count >= 0 && has_carry; --count) {
        if (end_buff[count] == '9') {
          end_buff[count] = '0';
        } else {
          end_buff[count] += 1;
          has_carry = false;
        }
      }
      // handle the high block that's buffered
      for (int count = buffered_digits - 1; count >= 0 && has_carry; --count) {
        if (block_buffer[count] == '9') {
          block_buffer[count] = '0';
        } else {
          block_buffer[count] += 1;
          has_carry = false;
        }
      }

      // has_carry should only be true here if every previous digit is 9, which
      // implies that the number has never been written.
      if (has_carry /* && !has_written */) {
        ++total_digits;
        ++digits_before_decimal;
        // Normally write_left_padding is called by flush_buffer but since we're
        // rounding up all of the digits, the ones in the buffer are wrong and
        // can't be flushed.
        RET_IF_RESULT_NEGATIVE(
            padding_writer.write_left_padding(writer, total_digits));
        // Now we know we need to print a leading 1, zeroes up to the decimal
        // point, the decimal point, and then finally digits after it.
        RET_IF_RESULT_NEGATIVE(writer->write('1'));
        // digits_before_decimal - 1 to account for the leading '1'
        RET_IF_RESULT_NEGATIVE(writer->write('0', digits_before_decimal - 1));
        if (has_decimal_point) {
          RET_IF_RESULT_NEGATIVE(writer->write(DECIMAL_POINT));
          // add one to digits_before_decimal to account for the decimal point
          // itself.
          if (total_digits > digits_before_decimal + 1) {
            RET_IF_RESULT_NEGATIVE(
                writer->write('0', total_digits - (digits_before_decimal + 1)));
          }
        }
        total_digits_written = total_digits;
        return 0;
      }
    }
    // Either we intend to round down, or the rounding up is complete. Flush the
    // buffers.

    RET_IF_RESULT_NEGATIVE(flush_buffer());

    // And then write the final block.
    RET_IF_RESULT_NEGATIVE(writer->write({end_buff, block_digits}));
    total_digits_written += block_digits;
    return 0;
  }

  int write_last_block_exp(uint32_t block, size_t block_digits,
                           RoundDirection round, int exponent, char exp_char) {
    char end_buff[BLOCK_SIZE];

    {
      char buf[IntegerToString::dec_bufsize<intmax_t>()];
      auto const int_to_str =
          *IntegerToString::dec(block + (MAX_BLOCK + 1), buf);

      // copy the last block_digits characters into the start of end_buff.
      // TODO: Replace with memcpy
      for (int count = block_digits - 1; count >= 0; --count) {
        end_buff[count] = int_to_str[count + 1 + (BLOCK_SIZE - block_digits)];
      }
    }

    char low_digit;
    if (block_digits > 0) {
      low_digit = end_buff[block_digits - 1];
    } else if (max_block_count > 0) {
      low_digit = '9';
    } else {
      low_digit = block_buffer[buffered_digits - 1];
    }

    // Round up
    if (round == RoundDirection::Up ||
        (round == RoundDirection::Even && low_digit % 2 != 0)) {
      bool has_carry = true;
      // handle the low block that we're adding
      for (int count = block_digits - 1; count >= 0 && has_carry; --count) {
        if (end_buff[count] == '9') {
          end_buff[count] = '0';
        } else {
          end_buff[count] += 1;
          has_carry = false;
        }
      }
      // handle the high block that's buffered
      for (int count = buffered_digits - 1; count >= 0 && has_carry; --count) {
        if (block_buffer[count] == '9') {
          block_buffer[count] = '0';
        } else {
          block_buffer[count] += 1;
          has_carry = false;
        }
      }

      // has_carry should only be true here if every previous digit is 9, which
      // implies that the number has never been written.
      if (has_carry /* && !has_written */) {
        // Since this is exponential notation, we don't write any more digits
        // but we do increment the exponent.
        ++exponent;

        char buf[IntegerToString::dec_bufsize<intmax_t>()];
        auto const int_to_str = exp_str(exponent, buf);

        // TODO: also change this to calculate the width of the number more
        // efficiently.
        size_t exponent_width = int_to_str.size();
        size_t number_digits =
            buffered_digits + (max_block_count * BLOCK_SIZE) + block_digits;

        // Here we have to recalculate the total number of digits since the
        // exponent's width may have changed. We're only adding 1 to exponent
        // width since exp_str appends the sign.
        total_digits =
            (has_decimal_point ? 1 : 0) + number_digits + 1 + exponent_width;

        // Normally write_left_padding is called by flush_buffer but since we're
        // rounding up all of the digits, the ones in the buffer are wrong and
        // can't be flushed.
        RET_IF_RESULT_NEGATIVE(
            padding_writer.write_left_padding(writer, total_digits));
        // Now we know we need to print a leading 1, the decimal point, and then
        // zeroes after it.
        RET_IF_RESULT_NEGATIVE(writer->write('1'));
        // digits_before_decimal - 1 to account for the leading '1'
        if (has_decimal_point) {
          RET_IF_RESULT_NEGATIVE(writer->write(DECIMAL_POINT));
          // This is just the length of the number, not including the decimal
          // point, or exponent.

          if (number_digits > 1) {
            RET_IF_RESULT_NEGATIVE(writer->write('0', number_digits - 1));
          }
        }
        RET_IF_RESULT_NEGATIVE(writer->write(exp_char));
        RET_IF_RESULT_NEGATIVE(writer->write(int_to_str));

        total_digits_written = total_digits;
        return WRITE_OK;
      }
    }
    // Either we intend to round down, or the rounding up is complete. Flush the
    // buffers.

    RET_IF_RESULT_NEGATIVE(flush_buffer());

    // And then write the final block. It's written via the buffer so that if
    // this is also the first block, the decimal point will be placed correctly.

    // TODO: Replace with memcpy
    for (size_t count = 0; count < block_digits; ++count) {
      block_buffer[count] = end_buff[count];
    }
    buffered_digits = block_digits;
    RET_IF_RESULT_NEGATIVE(flush_buffer());

    char buf[IntegerToString::dec_bufsize<intmax_t>()];
    auto const int_to_str = exp_str(exponent, buf);

    RET_IF_RESULT_NEGATIVE(writer->write(exp_char));
    RET_IF_RESULT_NEGATIVE(writer->write(int_to_str));

    total_digits_written = total_digits;

    return WRITE_OK;
  }

  int write_zeroes(uint32_t num_zeroes) {
    RET_IF_RESULT_NEGATIVE(flush_buffer());
    RET_IF_RESULT_NEGATIVE(writer->write('0', num_zeroes));
    return 0;
  }

  int right_pad() {
    return padding_writer.write_right_padding(writer, total_digits);
  }
};

// This implementation is based on the Ryu Printf algorithm by Ulf Adams:
// Ulf Adams. 2019. Ryū revisited: printf floating point conversion.
// Proc. ACM Program. Lang. 3, OOPSLA, Article 169 (October 2019), 23 pages.
// https://doi.org/10.1145/3360595
template <typename T, cpp::enable_if_t<cpp::is_floating_point_v<T>, int> = 0>
LIBC_INLINE int convert_float_decimal_typed(Writer *writer,
                                            const FormatSection &to_conv,
                                            fputil::FPBits<T> float_bits) {
  // signed because later we use -MANT_WIDTH
  constexpr int32_t MANT_WIDTH = fputil::MantissaWidth<T>::VALUE;
  bool is_negative = float_bits.get_sign();
  int exponent = float_bits.get_exponent();

  char sign_char = 0;

  if (is_negative)
    sign_char = '-';
  else if ((to_conv.flags & FormatFlags::FORCE_SIGN) == FormatFlags::FORCE_SIGN)
    sign_char = '+'; // FORCE_SIGN has precedence over SPACE_PREFIX
  else if ((to_conv.flags & FormatFlags::SPACE_PREFIX) ==
           FormatFlags::SPACE_PREFIX)
    sign_char = ' ';

  // If to_conv doesn't specify a precision, the precision defaults to 6.
  const size_t precision = to_conv.precision < 0 ? 6 : to_conv.precision;
  bool has_decimal_point =
      (precision > 0) || ((to_conv.flags & FormatFlags::ALTERNATE_FORM) != 0);

  // nonzero is false until a nonzero digit is found. It is used to determine if
  // leading zeroes should be printed, since before the first digit they are
  // ignored.
  bool nonzero = false;

  PaddingWriter padding_writer(to_conv, sign_char);
  FloatWriter float_writer(writer, has_decimal_point, padding_writer);
  FloatToString<T> float_converter(static_cast<T>(float_bits));

  const uint32_t positive_blocks = float_converter.get_positive_blocks();

  if (positive_blocks >= 0) {
    // This loop iterates through the number a block at a time until it finds a
    // block that is not zero or it hits the decimal point. This is because all
    // zero blocks before the first nonzero digit or the decimal point are
    // ignored (no leading zeroes, at least at this stage).
    int32_t i = static_cast<int32_t>(positive_blocks) - 1;
    for (; i >= 0; --i) {
      BlockInt digits = float_converter.get_positive_block(i);
      if (nonzero) {
        RET_IF_RESULT_NEGATIVE(float_writer.write_middle_block(digits));
      } else if (digits != 0) {
        size_t blocks_before_decimal = i;
        float_writer.init((blocks_before_decimal * BLOCK_SIZE) +
                              (has_decimal_point ? 1 : 0) + precision,
                          blocks_before_decimal * BLOCK_SIZE);
        float_writer.write_first_block(digits);

        nonzero = true;
      }
    }
  }

  // if we haven't yet found a valid digit, buffer a zero.
  if (!nonzero) {
    float_writer.init((has_decimal_point ? 1 : 0) + precision, 0);
    float_writer.write_first_block(0);
  }

  if (exponent < MANT_WIDTH) {
    const uint32_t blocks = (precision / BLOCK_SIZE) + 1;
    uint32_t i = 0;
    // if all the blocks we should write are zero
    if (blocks <= float_converter.zero_blocks_after_point()) {
      i = blocks; // just write zeroes up to precision
      RET_IF_RESULT_NEGATIVE(float_writer.write_zeroes(precision));
    } else if (i < float_converter.zero_blocks_after_point()) {
      // else if there are some blocks that are zeroes
      i = float_converter.zero_blocks_after_point();
      // write those blocks as zeroes.
      RET_IF_RESULT_NEGATIVE(float_writer.write_zeroes(9 * i));
    }
    // for each unwritten block
    for (; i < blocks; ++i) {
      if (float_converter.is_lowest_block(i)) {
        const uint32_t fill = precision - 9 * i;
        RET_IF_RESULT_NEGATIVE(float_writer.write_zeroes(fill));
        break;
      }
      BlockInt digits = float_converter.get_negative_block(i);
      if (i < blocks - 1) {
        RET_IF_RESULT_NEGATIVE(float_writer.write_middle_block(digits));
      } else {

        const uint32_t maximum = precision - BLOCK_SIZE * i;
        uint32_t last_digit = 0;
        for (uint32_t k = 0; k < BLOCK_SIZE - maximum; ++k) {
          last_digit = digits % 10;
          digits /= 10;
        }
        RoundDirection round;
        // Is m * 10^(additionalDigits + 1) / 2^(-exponent) integer?
        const int32_t requiredTwos =
            -(exponent - MANT_WIDTH) - static_cast<int32_t>(precision) - 1;
        const bool trailingZeros =
            requiredTwos <= 0 ||
            (requiredTwos < 60 &&
             multiple_of_power_of_2(float_bits.get_explicit_mantissa(),
                                    static_cast<uint32_t>(requiredTwos)));
        switch (fputil::quick_get_round()) {
        case FE_TONEAREST:
          // Round to nearest, if it's exactly halfway then round to even.
          if (last_digit != 5) {
            round = last_digit > 5 ? RoundDirection::Up : RoundDirection::Down;
          } else {
            round = trailingZeros ? RoundDirection::Even : RoundDirection::Up;
          }
          break;
        case FE_DOWNWARD:
          if (is_negative && (!trailingZeros || last_digit > 0)) {
            round = RoundDirection::Up;
          } else {
            round = RoundDirection::Down;
          }
          break;
        case FE_UPWARD:
          if (!is_negative && (!trailingZeros || last_digit > 0)) {
            round = RoundDirection::Up;
          } else {
            round = RoundDirection::Down;
          }
          round = is_negative ? RoundDirection::Down : RoundDirection::Up;
          break;
        case FE_TOWARDZERO:
          round = RoundDirection::Down;
          break;
        }
        RET_IF_RESULT_NEGATIVE(
            float_writer.write_last_block_dec(digits, maximum, round));
        break;
      }
    }
  } else {
    RET_IF_RESULT_NEGATIVE(float_writer.write_zeroes(precision));
  }
  RET_IF_RESULT_NEGATIVE(float_writer.right_pad());
  return WRITE_OK;
}

template <typename T, cpp::enable_if_t<cpp::is_floating_point_v<T>, int> = 0>
LIBC_INLINE int convert_float_dec_exp_typed(Writer *writer,
                                            const FormatSection &to_conv,
                                            fputil::FPBits<T> float_bits) {
  // signed because later we use -MANT_WIDTH
  constexpr int32_t MANT_WIDTH = fputil::MantissaWidth<T>::VALUE;
  bool is_negative = float_bits.get_sign();
  int exponent = float_bits.get_exponent();
  MantissaInt mantissa = float_bits.get_explicit_mantissa();

  const char a = (to_conv.conv_name & 32) | 'A';

  char sign_char = 0;

  if (is_negative)
    sign_char = '-';
  else if ((to_conv.flags & FormatFlags::FORCE_SIGN) == FormatFlags::FORCE_SIGN)
    sign_char = '+'; // FORCE_SIGN has precedence over SPACE_PREFIX
  else if ((to_conv.flags & FormatFlags::SPACE_PREFIX) ==
           FormatFlags::SPACE_PREFIX)
    sign_char = ' ';

  // If to_conv doesn't specify a precision, the precision defaults to 6.
  const size_t precision = to_conv.precision < 0 ? 6 : to_conv.precision;
  bool has_decimal_point =
      (precision > 0) || ((to_conv.flags & FormatFlags::ALTERNATE_FORM) != 0);

  PaddingWriter padding_writer(to_conv, sign_char);
  FloatWriter float_writer(writer, has_decimal_point, padding_writer);
  FloatToString<T> float_converter(static_cast<T>(float_bits));

  size_t digits_written = 0;
  int final_exponent = 0;

  // Here we would subtract 1 to account for the fact that block 0 counts as a
  // positive block, but the loop below accounts for this by starting with
  // subtracting 1 from cur_block.
  int cur_block;

  if (exponent < 0) {
    cur_block = -float_converter.zero_blocks_after_point();
  } else {
    cur_block = float_converter.get_positive_blocks();
  }

  BlockInt digits = 0;

  // If the mantissa is 0, then the number is 0, meaning that looping until a
  // non-zero block is found will loop forever. The first block is just 0.
  if (mantissa != 0) {
    // This loop finds the first block.
    while (digits == 0) {
      --cur_block;
      digits = float_converter.get_block(cur_block);
    }
  } else {
    cur_block = 0;
  }

  // TODO: Find a better way to calculate the number of digits in the
  // initial block and exponent.
  char buf[IntegerToString::dec_bufsize<intmax_t>()];
  auto int_to_str = *IntegerToString::dec(digits, buf);
  size_t block_width = int_to_str.size();

  final_exponent = (cur_block * BLOCK_SIZE) + (block_width - 1);
  int positive_exponent = final_exponent < 0 ? -final_exponent : final_exponent;

  int_to_str = *IntegerToString::dec(positive_exponent, buf);
  size_t exponent_width = int_to_str.size();

  // Calculate the total number of digits in the number.
  // 1 - the digit before the decimal point
  // 1 - the decimal point (optional)
  // precision - the number of digits after the decimal point
  // 1 - the 'e' at the start of the exponent
  // 1 - the sign at the start of the exponent
  // max(2, exp width) - the digits of the exponent, min 2.

  float_writer.init(1 + (has_decimal_point ? 1 : 0) + precision + 2 +
                        (exponent_width < 2 ? 2 : exponent_width),
                    1);

  // If this block is not the last block
  if (block_width <= precision + 1) {
    float_writer.write_first_block(digits, true);
    digits_written += block_width;
    --cur_block;
  }

  // For each middle block.
  for (; digits_written + BLOCK_SIZE < precision + 1; --cur_block) {
    digits = float_converter.get_block(cur_block);

    RET_IF_RESULT_NEGATIVE(float_writer.write_middle_block(digits));
    digits_written += BLOCK_SIZE;
  }

  digits = float_converter.get_block(cur_block);

  size_t last_block_size = BLOCK_SIZE;

  // if the last block is also the first block, then ignore leading zeroes.
  if (digits_written == 0) {
    // TODO: Find a better way to calculate the number of digits in a block.
    char buf[IntegerToString::dec_bufsize<intmax_t>()];
    auto int_to_str = *IntegerToString::dec(digits, buf);
    last_block_size = int_to_str.size();
  }

  // This is the last block.
  const uint32_t maximum = precision + 1 - digits_written;
  uint32_t last_digit = 0;
  for (uint32_t k = 0; k < last_block_size - maximum; ++k) {
    last_digit = digits % 10;
    digits /= 10;
  }

  // If the last block we read doesn't have the digit after the end of what
  // we'll print, then we need to read the next block to get that digit.
  if (maximum == last_block_size) {
    BlockInt extra_block = float_converter.get_block(cur_block - 1);
    last_digit = extra_block / ((MAX_BLOCK / 10) + 1);
  }

  RoundDirection round;
  // Is m * 10^(additionalDigits + 1) / 2^(-exponent) integer?
  const int32_t requiredTwos =
      -(exponent - MANT_WIDTH) - static_cast<int32_t>(precision) - 1;
  const bool trailingZeros =
      requiredTwos <= 0 ||
      (requiredTwos < 60 &&
       multiple_of_power_of_2(float_bits.get_explicit_mantissa(),
                              static_cast<uint32_t>(requiredTwos)));
  switch (fputil::quick_get_round()) {
  case FE_TONEAREST:
    // Round to nearest, if it's exactly halfway then round to even.
    if (last_digit != 5) {
      round = last_digit > 5 ? RoundDirection::Up : RoundDirection::Down;
    } else {
      round = trailingZeros ? RoundDirection::Even : RoundDirection::Up;
    }
    break;
  case FE_DOWNWARD:
    if (is_negative && (!trailingZeros || last_digit > 0)) {
      round = RoundDirection::Up;
    } else {
      round = RoundDirection::Down;
    }
    break;
  case FE_UPWARD:
    if (!is_negative && (!trailingZeros || last_digit > 0)) {
      round = RoundDirection::Up;
    } else {
      round = RoundDirection::Down;
    }
    round = is_negative ? RoundDirection::Down : RoundDirection::Up;
    break;
  case FE_TOWARDZERO:
    round = RoundDirection::Down;
    break;
  }
  RET_IF_RESULT_NEGATIVE(float_writer.write_last_block_exp(
      digits, maximum, round, final_exponent, a + 'E' - 'A'));

  RET_IF_RESULT_NEGATIVE(float_writer.right_pad());
  return WRITE_OK;
}

template <typename T, cpp::enable_if_t<cpp::is_floating_point_v<T>, int> = 0>
LIBC_INLINE int convert_float_dec_auto_typed(Writer *writer,
                                             const FormatSection &to_conv,
                                             fputil::FPBits<T> float_bits) {
  // signed because later we use -MANT_WIDTH
  constexpr int32_t MANT_WIDTH = fputil::MantissaWidth<T>::VALUE;
  bool is_negative = float_bits.get_sign();
  int exponent = float_bits.get_exponent();
  MantissaInt mantissa = float_bits.get_explicit_mantissa();

  // From the standard: Let P (init_precision) equal the precision if nonzero, 6
  // if the precision is omitted, or 1 if the precision is zero.
  const size_t init_precision = to_conv.precision <= 0
                                    ? (to_conv.precision == 0 ? 1 : 6)
                                    : to_conv.precision;

  //  Then, if a conversion with style E would have an exponent of X
  //  (base_10_exp):
  int base_10_exp = 0;
  // If P > X >= -4 the conversion is with style F and precision P - (X + 1).
  // Otherwise, the conversion is with style E and precision P - 1.

  // For calculating the base 10 exponent, we need to process the number as if
  // it has style E, so here we calculate the precision we'll use in that case.
  const size_t exp_precision = init_precision - 1;

  FloatToString<T> float_converter(static_cast<T>(float_bits));

  // Here we would subtract 1 to account for the fact that block 0 counts as a
  // positive block, but the loop below accounts for this by starting with
  // subtracting 1 from cur_block.
  int cur_block;

  if (exponent < 0) {
    cur_block = -float_converter.zero_blocks_after_point();
  } else {
    cur_block = float_converter.get_positive_blocks();
  }

  BlockInt digits = 0;

  // If the mantissa is 0, then the number is 0, meaning that looping until a
  // non-zero block is found will loop forever.
  if (mantissa != 0) {
    // This loop finds the first non-zero block.
    while (digits == 0) {
      --cur_block;
      digits = float_converter.get_block(cur_block);
    }
  } else {
    // In the case of 0.0, then it's always decimal format. If we don't have alt
    // form then the trailing zeroes are trimmed to make "0", else the precision
    // is 1 less than specified by the user.
    FormatSection new_conv = to_conv;
    if ((to_conv.flags & FormatFlags::ALTERNATE_FORM) != 0) {
      // This is a style F conversion, making the precision P - 1 - X, but since
      // this is for the number 0, X (the base 10 exponent) is always 0.
      new_conv.precision = init_precision - 1;
    } else {
      new_conv.precision = 0;
    }
    return convert_float_decimal_typed<T>(writer, new_conv, float_bits);
  }

  size_t block_width = 0;
  {
    // TODO: Find a better way to calculate the number of digits in the
    // initial block and exponent.
    char buf[IntegerToString::dec_bufsize<intmax_t>()];
    auto int_to_str = *IntegerToString::dec(digits, buf);
    block_width = int_to_str.size();
  }

  size_t digits_checked = 0;
  // TODO: look into unifying trailing_zeroes and trailing_nines. The number can
  // end in a nine or a zero, but not both.
  size_t trailing_zeroes = 0;
  size_t trailing_nines = 0;

  base_10_exp = (cur_block * BLOCK_SIZE) + (block_width - 1);

  // If the first block is not also the last block
  if (block_width <= exp_precision + 1) {
    char buf[IntegerToString::dec_bufsize<intmax_t>()];
    auto int_to_str = *IntegerToString::dec(digits, buf);

    for (size_t i = 0; i < block_width; ++i) {
      if (int_to_str[i] == '9') {
        ++trailing_nines;
        trailing_zeroes = 0;
      } else if (int_to_str[i] == '0') {
        ++trailing_zeroes;
        trailing_nines = 0;
      } else {
        trailing_nines = 0;
        trailing_zeroes = 0;
      }
    }
    digits_checked += block_width;
    --cur_block;
  }

  // Handle middle blocks
  for (; digits_checked + BLOCK_SIZE < exp_precision + 1; --cur_block) {
    digits = float_converter.get_block(cur_block);
    digits_checked += BLOCK_SIZE;
    if (digits == MAX_BLOCK) {
      trailing_nines += 9;
      trailing_zeroes = 0;
    } else if (digits == 0) {
      trailing_zeroes += 9;
      trailing_nines = 0;
    } else {
      // The block is neither all nines nor all zeroes, so we need to figure out
      // what it ends with.
      trailing_nines = 0;
      trailing_zeroes = 0;
      BlockInt copy_of_digits = digits;
      int cur_last_digit = copy_of_digits % 10;
      // We only care if it ends in nines or zeroes.
      while (copy_of_digits > 0 &&
             (cur_last_digit == 9 || cur_last_digit == 0)) {
        // If the next digit is not the same as the previous one, then there are
        // no more contiguous trailing digits.
        if ((copy_of_digits % 10) != cur_last_digit) {
          break;
        }
        if (cur_last_digit == 9) {
          ++trailing_nines;
        } else if (cur_last_digit == 0) {
          ++trailing_zeroes;
        } else {
          break;
        }
        copy_of_digits /= 10;
      }
    }
  }

  // Handle the last block

  digits = float_converter.get_block(cur_block);

  size_t last_block_size = BLOCK_SIZE;

  char buf[IntegerToString::dec_bufsize<intmax_t>()];
  auto int_to_str = *IntegerToString::dec(digits, buf);

  int implicit_leading_zeroes = BLOCK_SIZE - int_to_str.size();

  // if the last block is also the first block, then ignore leading zeroes.
  if (digits_checked == 0) {
    last_block_size = int_to_str.size();
    implicit_leading_zeroes = 0;
  } else {
    // If the block is not the maximum size, that means it has leading
    // zeroes, and zeroes are not nines.
    if (implicit_leading_zeroes > 0) {
      trailing_nines = 0;
    }

    // But leading zeroes are zeroes (that could be trailing).
    trailing_zeroes += implicit_leading_zeroes;
  }

  int digits_requested = (exp_precision + 1) - digits_checked;

  int digits_to_check = digits_requested - implicit_leading_zeroes;
  if (digits_to_check < 0) {
    digits_to_check = 0;
  }

  // Check the upper digits of this block.
  for (int i = 0; i < digits_to_check; ++i) {
    if (int_to_str[i] == '9') {
      ++trailing_nines;
      trailing_zeroes = 0;
    } else if (int_to_str[i] == '0') {
      ++trailing_zeroes;
      trailing_nines = 0;
    } else {
      trailing_nines = 0;
      trailing_zeroes = 0;
    }
  }

  // Find the digit after the lowest digit that we'll actually print to
  // determine the rounding.
  const uint32_t maximum = exp_precision + 1 - digits_checked;
  uint32_t last_digit = 0;
  for (uint32_t k = 0; k < last_block_size - maximum; ++k) {
    last_digit = digits % 10;
    digits /= 10;
  }

  // If the last block we read doesn't have the digit after the end of what
  // we'll print, then we need to read the next block to get that digit.
  if (maximum == last_block_size) {
    BlockInt extra_block = float_converter.get_block(cur_block - 1);
    last_digit = extra_block / ((MAX_BLOCK / 10) + 1);
  }

  // TODO: unify this code across the three float conversions.
  RoundDirection round;
  // Is m * 10^(additionalDigits + 1) / 2^(-exponent) integer?
  const int32_t requiredTwos =
      -(exponent - MANT_WIDTH) - static_cast<int32_t>(exp_precision) - 1;
  // TODO: rename this variable to remove confusion with trailing_zeroes
  const bool trailingZeros =
      requiredTwos <= 0 ||
      (requiredTwos < 60 &&
       multiple_of_power_of_2(float_bits.get_explicit_mantissa(),
                              static_cast<uint32_t>(requiredTwos)));
  switch (fputil::quick_get_round()) {
  case FE_TONEAREST:
    // Round to nearest, if it's exactly halfway then round to even.
    if (last_digit != 5) {
      round = last_digit > 5 ? RoundDirection::Up : RoundDirection::Down;
    } else {
      round = trailingZeros ? RoundDirection::Even : RoundDirection::Up;
    }
    break;
  case FE_DOWNWARD:
    if (is_negative && (!trailingZeros || last_digit > 0)) {
      round = RoundDirection::Up;
    } else {
      round = RoundDirection::Down;
    }
    break;
  case FE_UPWARD:
    if (!is_negative && (!trailingZeros || last_digit > 0)) {
      round = RoundDirection::Up;
    } else {
      round = RoundDirection::Down;
    }
    round = is_negative ? RoundDirection::Down : RoundDirection::Up;
    break;
  case FE_TOWARDZERO:
    round = RoundDirection::Down;
    break;
  }

  bool round_up;
  if (round == RoundDirection::Up) {
    round_up = true;
  } else if (round == RoundDirection::Down) {
    round_up = false;
  } else {
    // RoundDirection is even, so check the extra digit.
    uint32_t low_digit = digits % 10;
    round_up = (low_digit % 2) != 0;
  }

  digits_checked += digits_requested;
  LIBC_ASSERT(digits_checked == init_precision);
  // At this point we should have checked all the digits requested by the
  // precision. We may increment this number 1 more if we round up all of the
  // digits, but at this point in the code digits_checked should always equal
  // init_precision.

  if (round_up) {
    // If all the digits that would be printed are nines, then rounding up means
    // that the base 10 exponent is one higher and all those nines turn to
    // zeroes (e.g. 999 -> 1000).
    if (trailing_nines == init_precision) {
      ++base_10_exp;
      trailing_zeroes = digits_checked;
      ++digits_checked;
    } else {
      // If there are trailing nines, they turn into trailing zeroes when
      // they're rounded up.
      if (trailing_nines > 0) {
        trailing_zeroes += trailing_nines;
      } else if (trailing_zeroes > 0) {
        // If there are trailing zeroes, then the last digit will be rounded up
        // to a 1 so they aren't trailing anymore.
        trailing_zeroes = 0;
      }
    }
  }

  // if P > X >= -4, the conversion is with style f (or F) and precision equals
  //  P - (X + 1).
  if (static_cast<int>(init_precision) > base_10_exp && base_10_exp >= -4) {
    FormatSection new_conv = to_conv;
    const size_t conv_precision = init_precision - (base_10_exp + 1);

    if ((to_conv.flags & FormatFlags::ALTERNATE_FORM) != 0) {
      new_conv.precision = conv_precision;
    } else {
      // If alt form isn't set, then we need to determine the number of trailing
      // zeroes and set the precision such that they are removed.
      int trimmed_precision =
          digits_checked - (base_10_exp + 1) - trailing_zeroes;
      if (trimmed_precision < 0) {
        trimmed_precision = 0;
      }
      new_conv.precision =
          (static_cast<size_t>(trimmed_precision) > conv_precision)
              ? conv_precision
              : trimmed_precision;
    }

    return convert_float_decimal_typed<T>(writer, new_conv, float_bits);
  } else {
    // otherwise, the conversion is with style e (or E) and precision equals
    // P - 1
    const size_t conv_precision = init_precision - 1;
    FormatSection new_conv = to_conv;
    if ((to_conv.flags & FormatFlags::ALTERNATE_FORM) != 0) {
      new_conv.precision = conv_precision;
    } else {
      // If alt form isn't set, then we need to determine the number of trailing
      // zeroes and set the precision such that they are removed.
      int trimmed_precision = digits_checked - 1 - trailing_zeroes;
      if (trimmed_precision < 0) {
        trimmed_precision = 0;
      }
      new_conv.precision =
          (static_cast<size_t>(trimmed_precision) > conv_precision)
              ? conv_precision
              : trimmed_precision;
    }
    return convert_float_dec_exp_typed<T>(writer, new_conv, float_bits);
  }
}

// TODO: unify the float converters to remove the duplicated checks for inf/nan.
LIBC_INLINE int convert_float_decimal(Writer *writer,
                                      const FormatSection &to_conv) {
  if (to_conv.length_modifier == LengthModifier::L) {
    fputil::FPBits<long double>::UIntType float_raw = to_conv.conv_val_raw;
    fputil::FPBits<long double> float_bits(float_raw);
    if (!float_bits.is_inf_or_nan()) {
      return convert_float_decimal_typed<long double>(writer, to_conv,
                                                      float_bits);
    }
  } else {
    fputil::FPBits<double>::UIntType float_raw =
        static_cast<fputil::FPBits<double>::UIntType>(to_conv.conv_val_raw);
    fputil::FPBits<double> float_bits(float_raw);
    if (!float_bits.is_inf_or_nan()) {
      return convert_float_decimal_typed<double>(writer, to_conv, float_bits);
    }
  }

  return convert_inf_nan(writer, to_conv);
}

LIBC_INLINE int convert_float_dec_exp(Writer *writer,
                                      const FormatSection &to_conv) {
  if (to_conv.length_modifier == LengthModifier::L) {
    fputil::FPBits<long double>::UIntType float_raw = to_conv.conv_val_raw;
    fputil::FPBits<long double> float_bits(float_raw);
    if (!float_bits.is_inf_or_nan()) {
      return convert_float_dec_exp_typed<long double>(writer, to_conv,
                                                      float_bits);
    }
  } else {
    fputil::FPBits<double>::UIntType float_raw =
        static_cast<fputil::FPBits<double>::UIntType>(to_conv.conv_val_raw);
    fputil::FPBits<double> float_bits(float_raw);
    if (!float_bits.is_inf_or_nan()) {
      return convert_float_dec_exp_typed<double>(writer, to_conv, float_bits);
    }
  }

  return convert_inf_nan(writer, to_conv);
}

LIBC_INLINE int convert_float_dec_auto(Writer *writer,
                                       const FormatSection &to_conv) {
  if (to_conv.length_modifier == LengthModifier::L) {
    fputil::FPBits<long double>::UIntType float_raw = to_conv.conv_val_raw;
    fputil::FPBits<long double> float_bits(float_raw);
    if (!float_bits.is_inf_or_nan()) {
      return convert_float_dec_auto_typed<long double>(writer, to_conv,
                                                       float_bits);
    }
  } else {
    fputil::FPBits<double>::UIntType float_raw =
        static_cast<fputil::FPBits<double>::UIntType>(to_conv.conv_val_raw);
    fputil::FPBits<double> float_bits(float_raw);
    if (!float_bits.is_inf_or_nan()) {
      return convert_float_dec_auto_typed<double>(writer, to_conv, float_bits);
    }
  }

  return convert_inf_nan(writer, to_conv);
}

} // namespace printf_core
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_STDIO_PRINTF_CORE_FLOAT_DEC_CONVERTER_H
