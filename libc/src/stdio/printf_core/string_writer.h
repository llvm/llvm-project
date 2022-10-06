//===-- String Writer definition for printf ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_PRINTF_CORE_STRING_WRITER_H
#define LLVM_LIBC_SRC_STDIO_PRINTF_CORE_STRING_WRITER_H

#include "src/__support/CPP/string_view.h"
#include "src/string/memory_utils/memcpy_implementations.h"
#include <stddef.h>

namespace __llvm_libc {
namespace printf_core {

class StringWriter {
  char *__restrict cur_buffer;
  size_t available_capacity;

public:
  // StringWriter is intended to take a copy of the cur_buffer pointer, as well
  // as the maximum length of the string. This maximum length should not include
  // the null terminator, since that's written separately.
  StringWriter(char *__restrict buffer, size_t max_len = ~size_t(0))
      : cur_buffer(buffer), available_capacity(max_len) {}

  void write(cpp::string_view new_string);
  void write(char new_char, size_t len);
  void write(char new_char);

  // Terminate should only be called if the original max length passed to
  // snprintf was greater than 0. It writes a null byte to the end of the
  // cur_buffer, regardless of available_capacity.
  void terminate() { *cur_buffer = '\0'; }

  // These write functions take a StringWriter as a void* in raw_pointer, and
  // call the appropriate write function on it.
  static int write_str(void *raw_pointer, cpp::string_view new_string);
  static int write_chars(void *raw_pointer, char new_char, size_t len);
  static int write_char(void *raw_pointer, char new_char);
};

} // namespace printf_core
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_STDIO_PRINTF_CORE_STRING_WRITER_H
