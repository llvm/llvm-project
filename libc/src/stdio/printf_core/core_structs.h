//===-- Core Structures for printf ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_PRINTF_CORE_CORE_STRUCTS_H
#define LLVM_LIBC_SRC_STDIO_PRINTF_CORE_CORE_STRUCTS_H

#include <inttypes.h>
#include <stddef.h>

namespace __llvm_libc {
namespace printf_core {

// These length modifiers match the length modifiers in the format string, which
// is why they are formatted differently from the rest of the file.
enum class LengthModifier { hh, h, l, ll, j, z, t, L, none };

enum FormatFlags : uint8_t {
  LEFT_JUSTIFIED = 0x01, // -
  FORCE_SIGN = 0x02,     // +
  SPACE_PREFIX = 0x04,   // space
  ALTERNATE_FORM = 0x08, // #
  LEADING_ZEROES = 0x10, // 0

  // These flags come from the GNU extensions which aren't yet implemented.
  //  group_decimals = 0x20, // '
  //  locale_digits = 0x40,  // I
};

struct FormatSection {
  bool has_conv;

  const char *__restrict raw_string;
  size_t raw_len;

  // Format Specifier Values
  FormatFlags flags = FormatFlags(0);
  LengthModifier length_modifier = LengthModifier::none;
  int min_width = 0;
  int precision = -1;

  __uint128_t conv_val_raw; // Needs to be large enough to hold a long double.
  void *conv_val_ptr;

  char conv_name;
};

} // namespace printf_core
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_STDIO_PRINTF_CORE_CORE_STRUCTS_H
