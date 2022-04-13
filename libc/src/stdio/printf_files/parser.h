//===-- Format string parser for printf -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_PRINTF_FILES_PARSER_H
#define LLVM_LIBC_SRC_STDIO_PRINTF_FILES_PARSER_H

#include "src/stdio/printf_files/core_structs.h"

#include <stdarg.h>
#include <stddef.h>

namespace __llvm_libc {
namespace printf_core {

// TODO: Make this a compile option.
constexpr size_t TYPE_ARR_SIZE = 32;

class Parser {
  const char *__restrict str;

  size_t cur_pos = 0;

  va_list *vlist_start;
  va_list *vlist_cur;
  size_t vlist_index;

  // TODO: Make this an optional piece.
  VariableType type_arr[TYPE_ARR_SIZE];

  // TODO: Look into object stores for optimization.

public:
  Parser(const char *__restrict str, va_list *vlist);

  // get_next_section will parse the format string until it has a fully
  // specified format section. This can either be a raw format section with no
  // conversion, or a format section with a conversion that has all of its
  // variables stored in the format section.
  FormatSection get_next_section();

private:
  // get_arg_value gets the value from the vlist at index (starting at 1). This
  // may require parsing the format string. An index of 0 is interpreted as the
  // next value.
  template <class T> T get_arg_value(size_t index);
};

} // namespace printf_core
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_STDIO_PRINTF_FILES_PARSER_H
