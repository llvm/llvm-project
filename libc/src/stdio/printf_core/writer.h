//===-- String writer for printf --------------------------------*- C++ -*-===//
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

using WriteFunc = void (*)(void *, const char *__restrict, size_t);

class Writer final {
  // output is a pointer to the string or file that the writer is meant to write
  // to.
  void *output;

  // raw_write is a function that, when called on output with a char* and
  // length, will copy the number of bytes equal to the length from the char*
  // onto the end of output.
  WriteFunc raw_write;

  size_t max_length;
  size_t chars_written;

public:
  Writer(void *output, WriteFunc raw_write, size_t max_length);

  // write will copy length bytes from new_string into output using
  // raw_write, unless that would cause more bytes than max_length to be
  // written. It always increments chars_written by length.
  void write(const char *new_string, size_t length);

  // write_chars will copy length copies of new_char into output using raw_write
  // unless that would cause more bytes than max_length to be written. It always
  // increments chars_written by length.
  void write_chars(char new_char, size_t length);

  size_t get_chars_written();
};

} // namespace printf_core
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_STDIO_PRINTF_CORE_WRITER_H
