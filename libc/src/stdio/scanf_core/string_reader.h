//===-- String Reader definition for scanf ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_SCANF_CORE_STRING_READER_H
#define LLVM_LIBC_SRC_STDIO_SCANF_CORE_STRING_READER_H

#include <stddef.h>

namespace __llvm_libc {
namespace scanf_core {

class StringReader {
  const char *string;
  size_t cur_index = 0;

public:
  StringReader(const char *init_string) { string = init_string; }

  ~StringReader() {}

  char get_char();
  void unget_char();
};

} // namespace scanf_core
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_STDIO_SCANF_CORE_STRING_READER_H
