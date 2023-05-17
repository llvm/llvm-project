//===-- Reader definition for scanf -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_SCANF_CORE_READER_H
#define LLVM_LIBC_SRC_STDIO_SCANF_CORE_READER_H

#include "src/stdio/scanf_core/file_reader.h"
#include "src/stdio/scanf_core/string_reader.h"
#include <stddef.h>

namespace __llvm_libc {
namespace scanf_core {

enum class ReaderType { String, File };

class Reader final {
  union {
    StringReader *string_reader;
    FileReader *file_reader;
  };

  const ReaderType reader_type;

  size_t cur_chars_read = 0;

public:
  Reader(StringReader *init_string_reader)
      : string_reader(init_string_reader), reader_type(ReaderType::String) {}

  Reader(FileReader *init_file_reader)
      : file_reader(init_file_reader), reader_type(ReaderType::File) {}

  // This returns the next character from the input and advances it by one
  // character. When it hits the end of the string or file it returns '\0' to
  // signal to stop parsing.
  char getc();

  // This moves the input back by one character, placing c into the buffer if
  // this is a file reader, else c is ignored.
  void ungetc(char c);

  size_t chars_read() { return cur_chars_read; }

  bool has_error();
};

} // namespace scanf_core
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_STDIO_SCANF_CORE_READER_H
