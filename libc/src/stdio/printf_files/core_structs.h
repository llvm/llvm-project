//===-- Core Structures for printf ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_PRINTF_FILES_CORE_STRUCTS_H
#define LLVM_LIBC_SRC_STDIO_PRINTF_FILES_CORE_STRUCTS_H

#include <inttypes.h>
#include <stddef.h>

namespace __llvm_libc {
namespace printf_core {

enum class LengthModifier { hh, h, l, ll, j, z, t, L, none };
enum VariableType : uint8_t {
  // Types

  Void = 0x00,
  Char = 0x01,
  // WChar = 0x02,
  // WInt = 0x03,
  Short = 0x04,
  Int = 0x05,
  Long = 0x06,
  LLong = 0x07,
  Intmax = 0x08,
  Size = 0x09,
  Ptrdiff = 0x0a,
  Double = 0x0b,
  LDouble = 0x0c,

  // Modifiers

  Signed = 0x40,
  Pointer = 0x80,

  // Masks

  Type_Mask = 0x3f,
  Modifier_Mask = 0xc,
};

struct FormatSection {
  bool has_conv;

  const char *__restrict raw_string;
  size_t raw_len;

  // Format Specifier Values
  bool left_justified;
  bool force_sign;
  bool space_prefix;
  bool alt_form;
  bool leading_zeroes;
  LengthModifier length_modifier;
  int min_width;
  int precision;

  __uint128_t conv_val_raw; // Needs to be large enough to hold a long double.
  void *conv_val_ptr;

  char conv_name;
};

} // namespace printf_core
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_STDIO_PRINTF_FILES_CORE_STRUCTS_H
