//===-- Octal Converter for printf ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_PRINTF_CORE_OCT_CONVERTER_H
#define LLVM_LIBC_SRC_STDIO_PRINTF_CORE_OCT_CONVERTER_H

#include "src/stdio/printf_core/converter_utils.h"
#include "src/stdio/printf_core/core_structs.h"
#include "src/stdio/printf_core/writer.h"

#include <inttypes.h>
#include <stddef.h>

namespace __llvm_libc {
namespace printf_core {

int inline convert_oct(Writer *writer, const FormatSection &to_conv) {
  // This is the number of digits it takes to represent a octal value of a
  // certain number of bits. Each oct digit represents 3 bits, so the value is
  // ceil(number of bits / 3).
  constexpr size_t BUFF_LEN = ((sizeof(uintmax_t) * 8) + 2) / 3;
  uintmax_t num = to_conv.conv_val_raw;
  char buffer[BUFF_LEN];

  num = apply_length_modifier(num, to_conv.length_modifier);

  // Since the buffer is size to sized to be able fit the entire number, buf_cur
  // can never reach 0. So, we do not need bounds checking on buf_cur.
  size_t buff_cur = BUFF_LEN;
  for (; num > 0 /* && buff_cur > 0 */; --buff_cur, num /= 8)
    buffer[buff_cur - 1] = (num % 8) + '0';

  size_t num_digits = BUFF_LEN - buff_cur;

  // These are signed to prevent underflow due to negative values. Negative
  // values are treated the same as 0.
  int zeroes;
  int spaces;

  // Negative precision indicates that it was not specified.
  if (to_conv.precision < 0) {
    if ((to_conv.flags &
         (FormatFlags::LEADING_ZEROES | FormatFlags::LEFT_JUSTIFIED)) ==
        FormatFlags::LEADING_ZEROES) {
      // If this conv has flag 0 but not - and no specified precision, it's
      // padded with 0's instead of spaces identically to if precision =
      // min_width. For example: ("%04o", 15) -> "0017"
      zeroes = to_conv.min_width - num_digits;
      spaces = 0;
    } else if (num_digits < 1) {
      // If no precision is specified, precision defaults to 1. This means that
      // if the integer passed to the conversion is 0, a 0 will be printed.
      // Example: ("%3o", 0) -> "  0"
      zeroes = 1;
      spaces = to_conv.min_width - zeroes;
    } else {
      // If there are enough digits to pass over the precision, just write the
      // number, padded by spaces.
      zeroes = 0;
      spaces = to_conv.min_width - num_digits;
    }
  } else {
    // If precision was specified, possibly write zeroes, and possibly write
    // spaces. Example: ("%5.4o", 010000) -> "10000"
    // If the check for if zeroes is negative was not there, spaces would be
    // incorrectly evaluated as 1.
    zeroes = to_conv.precision - num_digits; // a negative value means 0
    if (zeroes < 0)
      zeroes = 0;
    spaces = to_conv.min_width - zeroes - num_digits;
  }

  // The alternate form prefix is "0", so it's handled by increasing the number
  // of zeroes if necessary.
  if (((to_conv.flags & FormatFlags::ALTERNATE_FORM) ==
       FormatFlags::ALTERNATE_FORM) &&
      zeroes < 1) {
    zeroes = 1;
    --spaces;
  }

  if ((to_conv.flags & FormatFlags::LEFT_JUSTIFIED) ==
      FormatFlags::LEFT_JUSTIFIED) {
    // If left justified the pattern is zeroes digits spaces
    if (zeroes > 0)
      RET_IF_RESULT_NEGATIVE(writer->write_chars('0', zeroes));
    if (num_digits > 0)
      RET_IF_RESULT_NEGATIVE(writer->write(buffer + buff_cur, num_digits));
    if (spaces > 0)
      RET_IF_RESULT_NEGATIVE(writer->write_chars(' ', spaces));
  } else {
    // Else the pattern is spaces zeroes digits
    if (spaces > 0)
      RET_IF_RESULT_NEGATIVE(writer->write_chars(' ', spaces));
    if (zeroes > 0)
      RET_IF_RESULT_NEGATIVE(writer->write_chars('0', zeroes));
    if (num_digits > 0)
      RET_IF_RESULT_NEGATIVE(writer->write(buffer + buff_cur, num_digits));
  }
  return 0;
}

} // namespace printf_core
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_STDIO_PRINTF_CORE_OCT_CONVERTER_H
