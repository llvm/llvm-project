//===-- FILE Writer definition for printf -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_PRINTF_CORE_FILE_WRITER_H
#define LLVM_LIBC_SRC_STDIO_PRINTF_CORE_FILE_WRITER_H

#include "src/__support/CPP/string_view.h"
#include "src/__support/File/file.h"

#include <stddef.h>
#include <stdio.h>

namespace __llvm_libc {
namespace printf_core {

class FileWriter {
  __llvm_libc::File *file;

public:
  FileWriter(::FILE *init_file) {
    file = reinterpret_cast<__llvm_libc::File *>(init_file);
    file->lock();
  }

  ~FileWriter() { file->unlock(); }

  int write(const char *__restrict to_write, size_t len);

  // These write functions take a FileWriter as a void* in raw_pointer, and
  // call the appropriate write function on it.
  static int write_str(void *raw_pointer, cpp::string_view new_string);
  static int write_chars(void *raw_pointer, char new_char, size_t len);
  static int write_char(void *raw_pointer, char new_char);
};

} // namespace printf_core
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_STDIO_PRINTF_CORE_FILE_WRITER_H
