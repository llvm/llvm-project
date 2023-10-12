//===-- Core Structures for printf ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_PRINTF_CORE_CORE_STRUCTS_H
#define LLVM_LIBC_SRC_STDIO_PRINTF_CORE_CORE_STRUCTS_H

#include "src/__support/CPP/string_view.h"
#include "src/__support/FPUtil/FPBits.h"

#include <inttypes.h>
#include <stddef.h>

namespace LIBC_NAMESPACE {
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

  cpp::string_view raw_string;

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
  bool operator==(const FormatSection &other) const {
    if (has_conv != other.has_conv)
      return false;

    if (raw_string != other.raw_string)
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

enum PrimaryType : uint8_t { Unknown = 0, Float = 1, Pointer = 2, Integer = 3 };

// TypeDesc stores the information about a type that is relevant to printf in
// a relatively compact manner.
struct TypeDesc {
  uint8_t size;
  PrimaryType primary_type;
  LIBC_INLINE constexpr bool operator==(const TypeDesc &other) const {
    return (size == other.size) && (primary_type == other.primary_type);
  }
};

template <typename T> LIBC_INLINE constexpr TypeDesc type_desc_from_type() {
  if constexpr (cpp::is_same_v<T, void>) {
    return TypeDesc{0, PrimaryType::Unknown};
  } else {
    constexpr bool isPointer = cpp::is_pointer_v<T>;
    constexpr bool isFloat = cpp::is_floating_point_v<T>;
    return TypeDesc{sizeof(T), isPointer ? PrimaryType::Pointer
                               : isFloat ? PrimaryType::Float
                                         : PrimaryType::Integer};
  }
}

// This is the value to be returned by conversions when no error has occurred.
constexpr int WRITE_OK = 0;
// These are the printf return values for when an error has occurred. They are
// all negative, and should be distinct.
constexpr int FILE_WRITE_ERROR = -1;
constexpr int FILE_STATUS_ERROR = -2;
constexpr int NULLPTR_WRITE_ERROR = -3;
constexpr int INT_CONVERSION_ERROR = -4;

} // namespace printf_core
} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_STDIO_PRINTF_CORE_CORE_STRUCTS_H
