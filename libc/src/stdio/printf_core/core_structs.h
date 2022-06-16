//===-- Core Structures for printf ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_PRINTF_CORE_CORE_STRUCTS_H
#define LLVM_LIBC_SRC_STDIO_PRINTF_CORE_CORE_STRUCTS_H

#include "src/__support/CPP/StringView.h"
#include "src/__support/FPUtil/FPBits.h"

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

  // Needs to be large enough to hold a long double.
  fputil::FPBits<long double>::UIntType conv_val_raw;
  void *conv_val_ptr;

  char conv_name;

  // This operator is only used for testing and should be automatically
  // optimized out for release builds.
  bool operator==(const FormatSection &other) {
    if (has_conv != other.has_conv)
      return false;

    if (!cpp::StringView(raw_string, raw_len)
             .equals(cpp::StringView(other.raw_string, other.raw_len)))
      return false;

    if (has_conv) {
      if (!((static_cast<uint8_t>(flags) ==
             static_cast<uint8_t>(other.flags)) &&
            (min_width == other.min_width) && (precision == other.precision) &&
            (length_modifier == other.length_modifier) &&
            (conv_name == other.conv_name)))
        return false;

      if (conv_name == 'p' || conv_name == 'n' || conv_name == 's')
        return (conv_val_ptr == other.conv_val_ptr);
      else if (conv_name != '%')
        return (conv_val_raw == other.conv_val_raw);
    }
    return true;
  }
};

#define RET_IF_RESULT_NEGATIVE(func)                                           \
  {                                                                            \
    int result = (func);                                                       \
    if (result < 0)                                                            \
      return result;                                                           \
  }

} // namespace printf_core
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_STDIO_PRINTF_CORE_CORE_STRUCTS_H
