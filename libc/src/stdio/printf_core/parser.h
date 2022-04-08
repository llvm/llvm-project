//===-- Format string parser for printf -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_PRINTF_CORE_PARSER_H
#define LLVM_LIBC_SRC_STDIO_PRINTF_CORE_PARSER_H

#include "src/__support/arg_list.h"
#include "src/stdio/printf_core/core_structs.h"

#include <stddef.h>

namespace __llvm_libc {
namespace printf_core {

class Parser {
  const char *__restrict str;

  size_t cur_pos = 0;

  internal::ArgList args_start;
  internal::ArgList args_cur;
  size_t args_index = 1;

  // TODO: Look into object stores for optimization.

public:
  Parser(const char *__restrict new_str, internal::ArgList &args)
      : str(new_str), args_start(args), args_cur(args) {}

  // get_next_section will parse the format string until it has a fully
  // specified format section. This can either be a raw format section with no
  // conversion, or a format section with a conversion that has all of its
  // variables stored in the format section.
  FormatSection get_next_section();

private:
  // parse_flags parses the flags inside a format string. It assumes that
  // str[*local_pos] is inside a format specifier, and parses any flags it
  // finds. It returns a FormatFlags object containing the set of found flags
  // arithmetically or'd together. local_pos will be moved past any flags found.
  FormatFlags parse_flags(size_t *local_pos);

  // parse_length_modifier parses the length modifier inside a format string. It
  // assumes that str[*local_pos] is inside a format specifier. It returns a
  // LengthModifier with the length modifier it found. It will advance local_pos
  // after the format specifier if one is found.
  LengthModifier parse_length_modifier(size_t *local_pos);

  // get_next_arg_value gets the next value from the arg list as type T.
  template <class T> T inline get_next_arg_value() {
    ++args_index;
    return args_cur.next_var<T>();
  }
};

} // namespace printf_core
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_STDIO_PRINTF_CORE_PARSER_H
