//===-- Integer Converter for printf ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_PRINTF_CORE_INT_CONVERTER_H
#define LLVM_LIBC_SRC_STDIO_PRINTF_CORE_INT_CONVERTER_H

#include "src/__support/integer_to_string.h"
#include "src/stdio/printf_core/converter_utils.h"
#include "src/stdio/printf_core/core_structs.h"
#include "src/stdio/printf_core/writer.h"

#include <inttypes.h>
#include <stddef.h>

namespace __llvm_libc {
namespace printf_core {

int inline convert_int(Writer *writer, const FormatSection &to_conv) {
  static constexpr size_t BITS_IN_BYTE = 8;
  static constexpr size_t BITS_IN_NUM = sizeof(uintmax_t) * BITS_IN_BYTE;

  uintmax_t num = to_conv.conv_val_raw;
  bool is_negative = false;
  FormatFlags flags = to_conv.flags;

  if (to_conv.conv_name == 'u') {
    // These flags are only for signed conversions, so this removes them if the
    // conversion is unsigned.
    flags = FormatFlags(flags &
                        ~(FormatFlags::FORCE_SIGN | FormatFlags::SPACE_PREFIX));
  } else {
    // Check if the number is negative by checking the high bit. This works even
    // for smaller numbers because they're sign extended by default.
    if ((num & (uintmax_t(1) << (BITS_IN_NUM - 1))) > 0) {
      is_negative = true;
      num = -num;
    }
  }

  num = apply_length_modifier(num, to_conv.length_modifier);

  auto const int_to_str = integer_to_string(num);
  size_t digits_written = int_to_str.str().size();

  char sign_char = 0;

  if (is_negative)
    sign_char = '-';
  else if ((flags & FormatFlags::FORCE_SIGN) == FormatFlags::FORCE_SIGN)
    sign_char = '+'; // FORCE_SIGN has precedence over SPACE_PREFIX
  else if ((flags & FormatFlags::SPACE_PREFIX) == FormatFlags::SPACE_PREFIX)
    sign_char = ' ';

  int sign_char_len = (sign_char == 0 ? 0 : 1);

  // These are signed to prevent underflow due to negative values. The eventual
  // values will always be non-negative.
  int zeroes;
  int spaces;

  // Negative precision indicates that it was not specified.
  if (to_conv.precision < 0) {
    if ((flags & (FormatFlags::LEADING_ZEROES | FormatFlags::LEFT_JUSTIFIED)) ==
        FormatFlags::LEADING_ZEROES) {
      // If this conv has flag 0 but not - and no specified precision, it's
      // padded with 0's instead of spaces identically to if precision =
      // min_width - (1 if sign_char). For example: ("%+04d", 1) -> "+001"
      zeroes = to_conv.min_width - digits_written - sign_char_len;
      if (zeroes < 0)
        zeroes = 0;
      spaces = 0;
    } else {
      // If there are enough digits to pass over the precision, just write the
      // number, padded by spaces.
      zeroes = 0;
      spaces = to_conv.min_width - digits_written - sign_char_len;
    }
  } else {
    // If precision was specified, possibly write zeroes, and possibly write
    // spaces. Example: ("%5.4d", 10000) -> "10000"
    // If the check for if zeroes is negative was not there, spaces would be
    // incorrectly evaluated as 1.
    //
    // The standard treats the case when num and precision are both zeroes as
    // special - it requires that no characters are produced. So, we adjust for
    // that special case first.
    if (num == 0 && to_conv.precision == 0)
      digits_written = 0;
    zeroes = to_conv.precision - digits_written; // a negative value means 0
    if (zeroes < 0)
      zeroes = 0;
    spaces = to_conv.min_width - zeroes - digits_written - sign_char_len;
  }
  if (spaces < 0)
    spaces = 0;

  if ((flags & FormatFlags::LEFT_JUSTIFIED) == FormatFlags::LEFT_JUSTIFIED) {
    // If left justified it goes sign zeroes digits spaces
    if (sign_char != 0)
      RET_IF_RESULT_NEGATIVE(writer->write(&sign_char, 1));
    if (zeroes > 0)
      RET_IF_RESULT_NEGATIVE(writer->write_chars('0', zeroes));
    if (digits_written > 0)
      RET_IF_RESULT_NEGATIVE(
          writer->write(int_to_str.str().data(), digits_written));
    if (spaces > 0)
      RET_IF_RESULT_NEGATIVE(writer->write_chars(' ', spaces));
  } else {
    // Else it goes spaces sign zeroes digits
    if (spaces > 0)
      RET_IF_RESULT_NEGATIVE(writer->write_chars(' ', spaces));
    if (sign_char != 0)
      RET_IF_RESULT_NEGATIVE(writer->write(&sign_char, 1));
    if (zeroes > 0)
      RET_IF_RESULT_NEGATIVE(writer->write_chars('0', zeroes));
    if (digits_written > 0)
      RET_IF_RESULT_NEGATIVE(
          writer->write(int_to_str.str().data(), digits_written));
  }
  return WRITE_OK;
}

} // namespace printf_core
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_STDIO_PRINTF_CORE_INT_CONVERTER_H
