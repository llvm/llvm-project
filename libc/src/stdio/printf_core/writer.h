//===-- Writer definition for printf ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_PRINTF_CORE_WRITER_H
#define LLVM_LIBC_SRC_STDIO_PRINTF_CORE_WRITER_H

#include "src/__support/CPP/string_view.h"
#include <stddef.h>

namespace __llvm_libc {
namespace printf_core {

using WriteStrFunc = int (*)(void *, cpp::string_view);
using WriteCharsFunc = int (*)(void *, char, size_t);
using WriteCharFunc = int (*)(void *, char);

class Writer final {
  // output is a pointer to the string or file that the writer is meant to write
  // to.
  void *output;

  // raw_write is a function that, when called on output with a char* and
  // length, will copy the number of bytes equal to the length from the char*
  // onto the end of output. It should return a positive number or zero on
  // success, or a negative number on failure.
  WriteStrFunc str_write;
  WriteCharsFunc chars_write;
  WriteCharFunc char_write;

  int chars_written = 0;

public:
  Writer(void *init_output, WriteStrFunc init_str_write,
         WriteCharsFunc init_chars_write, WriteCharFunc init_char_write)
      : output(init_output), str_write(init_str_write),
        chars_write(init_chars_write), char_write(init_char_write) {}

  // write will copy new_string into output using str_write. It increments
  // chars_written by the length of new_string. It returns the result of
  // str_write.
  int write(cpp::string_view new_string);

  // this version of write will copy length copies of new_char into output using
  // chars_write. This is primarily used for padding.  It returns the result of
  // chars_write.
  int write(char new_char, size_t len);

  // this version of write will copy just new_char into output. This is often
  // used for negative signs. It returns the result of chars_write.
  int write(char new_char);

  int get_chars_written() { return chars_written; }
};

} // namespace printf_core
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_STDIO_PRINTF_CORE_WRITER_H
