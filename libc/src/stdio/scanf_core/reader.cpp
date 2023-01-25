//===-- Reader definition for scanf -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/scanf_core/reader.h"
#include <stddef.h>

namespace __llvm_libc {
namespace scanf_core {

char Reader::getc() {
  ++cur_chars_read;
  if (reader_type == ReaderType::String) {
    return string_reader->get_char();
  } else {
    return file_reader->get_char();
  }
}

void Reader::ungetc(char c) {
  --cur_chars_read;
  if (reader_type == ReaderType::String) {
    // The string reader ignores the char c passed to unget since it doesn't
    // need to place anything back into a buffer, and modifying the source
    // string would be dangerous.
    return string_reader->unget_char();
  } else {
    return file_reader->unget_char(c);
  }
}

bool Reader::has_error() {
  if (reader_type == ReaderType::File) {
    return file_reader->has_error();
  }
  return false;
}

} // namespace scanf_core
} // namespace __llvm_libc
