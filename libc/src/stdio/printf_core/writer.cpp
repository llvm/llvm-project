//===-- Writer definition for printf ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "writer.h"
#include "src/string/memory_utils/memset_implementations.h"
#include <stddef.h>

namespace __llvm_libc {
namespace printf_core {

void Writer::write(const char *new_string, size_t length) {

  raw_write(output, new_string, length);

  // chars_written tracks the number of chars that would have been written
  // regardless of what the raw_write call does.
  chars_written += length;
}

void Writer::write_chars(char new_char, size_t length) {
  constexpr size_t BUFF_SIZE = 8;
  char buff[BUFF_SIZE];
  inline_memset(buff, new_char, BUFF_SIZE);
  while (length > BUFF_SIZE) {
    write(buff, BUFF_SIZE);
    length -= BUFF_SIZE;
  }
  write(buff, length);
}

} // namespace printf_core
} // namespace __llvm_libc
