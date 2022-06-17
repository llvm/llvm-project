//===-- Hexadecimal Converter for printf ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_PRINTF_CORE_HEX_CONVERTER_H
#define LLVM_LIBC_SRC_STDIO_PRINTF_CORE_HEX_CONVERTER_H

#include "src/stdio/printf_core/converter_utils.h"
#include "src/stdio/printf_core/core_structs.h"
#include "src/stdio/printf_core/writer.h"

#include <inttypes.h>
#include <stddef.h>

namespace __llvm_libc {
namespace printf_core {

int convert_hex(Writer *writer, const FormatSection &to_conv) {
  // This approximates the number of digits it takes to represent a hexadecimal
  // value of a certain number of bits. Each hex digit represents 4 bits, so the
  // exact value is the number of bytes multiplied by 2.
  static constexpr size_t BUFF_LEN = sizeof(uintmax_t) * 2;
  uintmax_t num = to_conv.conv_val_raw;
  char buffer[BUFF_LEN];

  // All of the characters will be defined relative to variable a, which will be
  // the appropriate case based on the name of the conversion.
  char a;
  if (to_conv.conv_name == 'x')
    a = 'a';
  else
    a = 'A';

  num = apply_length_modifier(num, to_conv.length_modifier);

  // buff_cur can never reach 0, since the buffer is sized to always be able to
  // contain the whole integer. This means that bounds checking it should be
  // unnecessary.
  size_t buff_cur = BUFF_LEN;
  for (; num > 0 /* && buff_cur > 0 */; --buff_cur, num /= 16)
    buffer[buff_cur - 1] =
        ((num % 16) > 9) ? ((num % 16) - 10 + a) : ((num % 16) + '0');

  size_t digits_written = BUFF_LEN - buff_cur;

  // these are signed to prevent underflow due to negative values. The eventual
  // values will always be non-negative.
  int zeroes;
  int spaces;

  // prefix is "0x"
  int prefix_len;
  char prefix[2];
  if ((to_conv.flags & FormatFlags::ALTERNATE_FORM) ==
      FormatFlags::ALTERNATE_FORM) {
    prefix_len = 2;
    prefix[0] = '0';
    prefix[1] = a + ('x' - 'a');
  } else {
    prefix_len = 0;
    prefix[0] = 0;
  }

  // negative precision indicates that it was not specified.
  if (to_conv.precision < 0) {
    if ((to_conv.flags &
         (FormatFlags::LEADING_ZEROES | FormatFlags::LEFT_JUSTIFIED)) ==
        FormatFlags::LEADING_ZEROES) {
      // if this conv has flag 0 but not - and no specified precision, it's
      // padded with 0's instead of spaces identically to if precision =
      // min_width - (2 if prefix). For example: ("%#04x", 15) -> "0x0f"
      zeroes = to_conv.min_width - digits_written - prefix_len;
      if (zeroes < 0)
        zeroes = 0;
      spaces = 0;
    } else if (digits_written < 1) {
      // if no precision is specified, precision defaults to 1. This means that
      // if the integer passed to the conversion is 0, a 0 will be printed.
      // Example: ("%3x", 0) -> "  0"
      zeroes = 1;
      spaces = to_conv.min_width - zeroes - prefix_len;
    } else {
      // If there are enough digits to pass over the precision, just write the
      // number, padded by spaces.
      zeroes = 0;
      spaces = to_conv.min_width - digits_written - prefix_len;
    }
  } else {
    // if precision was specified, possibly write zeroes, and possibly write
    // spaces. Example: ("%5.4x", 0x10000) -> "10000"
    // If the check for if zeroes is negative was not there, spaces would be
    // incorrectly evaluated as 1.
    zeroes = to_conv.precision - digits_written; // a negative value means 0
    if (zeroes < 0)
      zeroes = 0;
    spaces = to_conv.min_width - zeroes - digits_written - prefix_len;
  }
  if (spaces < 0)
    spaces = 0;

  if ((to_conv.flags & FormatFlags::LEFT_JUSTIFIED) ==
      FormatFlags::LEFT_JUSTIFIED) {
    // if left justified it goes prefix zeroes digits spaces
    if (prefix[0] != 0)
      RET_IF_RESULT_NEGATIVE(writer->write(prefix, 2));
    if (zeroes > 0)
      RET_IF_RESULT_NEGATIVE(writer->write_chars('0', zeroes));
    if (digits_written > 0)
      RET_IF_RESULT_NEGATIVE(writer->write(buffer + buff_cur, digits_written));
    if (spaces > 0)
      RET_IF_RESULT_NEGATIVE(writer->write_chars(' ', spaces));
  } else {
    // else it goes spaces prefix zeroes digits
    if (spaces > 0)
      RET_IF_RESULT_NEGATIVE(writer->write_chars(' ', spaces));
    if (prefix[0] != 0)
      RET_IF_RESULT_NEGATIVE(writer->write(prefix, 2));
    if (zeroes > 0)
      RET_IF_RESULT_NEGATIVE(writer->write_chars('0', zeroes));
    if (digits_written > 0)
      RET_IF_RESULT_NEGATIVE(writer->write(buffer + buff_cur, digits_written));
  }
  return 0;
}

} // namespace printf_core
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_STDIO_PRINTF_CORE_HEX_CONVERTER_H
