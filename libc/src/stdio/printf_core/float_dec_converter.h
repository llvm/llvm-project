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
#include "src/__support/UInt.h"
#include "src/__support/UInt128.h"
#include "src/__support/float_to_string.h"
#include "src/__support/integer_to_string.h"
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
constexpr inline bool multiple_of_power_of_2(const uint64_t value,
                                             const uint32_t p) {
  return (value & ((uint64_t(1) << p) - 1)) == 0;
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
  char block_buffer[BLOCK_SIZE];
  size_t buffered_digits = 0;
  bool has_written = false;
  size_t max_block_count = 0;
  size_t total_digits = 0;
  size_t digits_before_decimal = 0;
  size_t total_digits_written = 0;
  bool has_decimal_point;
  Writer *writer;
  PaddingWriter padding_writer;

  int flush_buffer() {
    // Write the most recent buffered block, and mark has_written
    if (!has_written) {
      has_written = true;
      RET_IF_RESULT_NEGATIVE(
          padding_writer.write_left_padding(writer, total_digits));
    }

    if (buffered_digits > 0) {
      RET_IF_RESULT_NEGATIVE(writer->write({block_buffer, buffered_digits}));
      total_digits_written += buffered_digits;
      buffered_digits = 0;
    }

    // if the decimal point is the next character, or is in the range covered
    // by the max blocks, write the appropriate digits and the decimal point.
    if (total_digits_written <= digits_before_decimal &&
        total_digits_written + BLOCK_SIZE * max_block_count >=
            digits_before_decimal &&
        has_decimal_point) {
      size_t digits_to_write = digits_before_decimal - total_digits_written;
      if (digits_to_write > 0) {
        RET_IF_RESULT_NEGATIVE(writer->write('9', digits_to_write));
      }
      RET_IF_RESULT_NEGATIVE(writer->write(DECIMAL_POINT));
      if (digits_to_write - (BLOCK_SIZE * max_block_count) > 0) {
        RET_IF_RESULT_NEGATIVE(writer->write(
            '9', digits_to_write - (BLOCK_SIZE * max_block_count)));
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

public:
  FloatWriter(Writer *init_writer, bool init_has_decimal_point,
              const PaddingWriter &init_padding_writer)
      : has_decimal_point(init_has_decimal_point), writer(init_writer),
        padding_writer(init_padding_writer) {}

  void init(size_t init_total_digits, size_t init_digits_before_decimal) {
    total_digits = init_total_digits;
    digits_before_decimal = init_digits_before_decimal;
  }

  void write_first_block(BlockInt block) {
    char buf[IntegerToString::dec_bufsize<intmax_t>()];
    auto const int_to_str = *IntegerToString::dec(block, buf);
    size_t digits_written = int_to_str.size();
    // Block Buffer is guaranteed to not overflow since block cannot have more
    // than BLOCK_SIZE digits.
    // TODO: Replace with memcpy
    for (size_t count = 0; count < digits_written; ++count) {
      block_buffer[count] = int_to_str[count];
    }
    total_digits_written = 0;
    buffered_digits = digits_written;
    total_digits += digits_written;
    digits_before_decimal += digits_written;
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
        // TODO: Handle prefixes here
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

  int write_last_block_exp(uint32_t block, size_t block_digits, int exponent) {
    // TODO
    //  This should be almost identical to the above, except in the case of
    //  rounding all digits up. Instead of adding an extra digit in front of the
    //  decimal point, we want to add 1 to the exponent.
    // Also we need to write the exponent, but that's pretty simple.
    return -1;
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

template <typename T, cpp::enable_if_t<cpp::is_floating_point_v<T>, int> = 0>
int inline convert_float_decimal_typed(Writer *writer,
                                       const FormatSection &to_conv,
                                       fputil::FPBits<T> float_bits) {
  // signed because later we use -MANT_WIDTH
  constexpr int32_t MANT_WIDTH = fputil::MantissaWidth<T>::VALUE;
  bool is_negative = float_bits.get_sign();
  int exponent = float_bits.get_exponent();
  MantissaInt mantissa = float_bits.get_explicit_mantissa();

  char sign_char = 0;

  if (is_negative)
    sign_char = '-';
  else if ((to_conv.flags & FormatFlags::FORCE_SIGN) == FormatFlags::FORCE_SIGN)
    sign_char = '+'; // FORCE_SIGN has precedence over SPACE_PREFIX
  else if ((to_conv.flags & FormatFlags::SPACE_PREFIX) ==
           FormatFlags::SPACE_PREFIX)
    sign_char = ' ';

  // If to_conv doesn't specify a precision, the precision defaults to 6.
  size_t precision = to_conv.precision < 0 ? 6 : to_conv.precision;
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
        uint32_t lastDigit = 0;
        for (uint32_t k = 0; k < BLOCK_SIZE - maximum; ++k) {
          lastDigit = digits % 10;
          digits /= 10;
        }
        RoundDirection round;
        // Is m * 10^(additionalDigits + 1) / 2^(-exponent) integer?
        const int32_t requiredTwos =
            -exponent - MANT_WIDTH - (int32_t)precision - 1;
        const bool trailingZeros =
            requiredTwos <= 0 ||
            (requiredTwos < 60 &&
             multiple_of_power_of_2(float_bits.get_explicit_mantissa(),
                                    (uint32_t)requiredTwos));
        switch (fputil::get_round()) {
        case FE_TONEAREST:
          // Round to nearest, if it's exactly halfway then round to even.
          if (lastDigit != 5) {
            round = lastDigit > 5 ? RoundDirection::Up : RoundDirection::Down;
          } else {
            round = trailingZeros ? RoundDirection::Even : RoundDirection::Up;
          }
          break;
        case FE_DOWNWARD:
          if (is_negative && (!trailingZeros || lastDigit > 0)) {
            round = RoundDirection::Up;
          } else {
            round = RoundDirection::Down;
          }
          break;
        case FE_UPWARD:
          if (!is_negative && (!trailingZeros || lastDigit > 0)) {
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

int inline convert_float_decimal(Writer *writer, const FormatSection &to_conv) {
  if (to_conv.length_modifier == LengthModifier::L) {
    fputil::FPBits<long double>::UIntType float_raw = to_conv.conv_val_raw;
    fputil::FPBits<long double> float_bits(float_raw);
    if (!float_bits.is_inf_or_nan()) {
      return convert_float_decimal_typed<long double>(writer, to_conv,
                                                      float_bits);
    }
  } else {
    fputil::FPBits<double>::UIntType float_raw = to_conv.conv_val_raw;
    fputil::FPBits<double> float_bits(float_raw);
    if (!float_bits.is_inf_or_nan()) {
      return convert_float_decimal_typed<double>(writer, to_conv, float_bits);
    }
  }

  return convert_inf_nan(writer, to_conv);
}
} // namespace printf_core
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_STDIO_PRINTF_CORE_FLOAT_DEC_CONVERTER_H
