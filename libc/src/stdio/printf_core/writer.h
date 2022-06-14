//===-- Writer definition for printf ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_PRINTF_CORE_WRITER_H
#define LLVM_LIBC_SRC_STDIO_PRINTF_CORE_WRITER_H

#include <stddef.h>

namespace __llvm_libc {
namespace printf_core {

using WriteFunc = int (*)(void *, const char *__restrict, size_t);

class Writer final {
  // output is a pointer to the string or file that the writer is meant to write
  // to.
  void *output;

  // raw_write is a function that, when called on output with a char* and
  // length, will copy the number of bytes equal to the length from the char*
  // onto the end of output. It should return a positive number or zero on
  // success, or a negative number on failure.
  WriteFunc raw_write;

  int chars_written = 0;

public:
  Writer(void *init_output, WriteFunc init_raw_write)
      : output(init_output), raw_write(init_raw_write) {}

  // write will copy length bytes from new_string into output using
  // raw_write. It increments chars_written by length always. It returns the
  // result of raw_write.
  int write(const char *new_string, size_t length);

  // write_chars will copy length copies of new_char into output using the write
  // function and a statically sized buffer. This is primarily used for padding.
  // If write returns a negative value, this will return early with that value.
  int write_chars(char new_char, size_t length);

  int get_chars_written() { return chars_written; }
};

} // namespace printf_core
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_STDIO_PRINTF_CORE_WRITER_H
