//===-- FILE Reader definition for scanf ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_SCANF_CORE_FILE_READER_H
#define LLVM_LIBC_SRC_STDIO_SCANF_CORE_FILE_READER_H

#include "src/__support/File/file.h"

#include <stddef.h>
#include <stdio.h>

namespace __llvm_libc {
namespace scanf_core {

class FileReader {
  __llvm_libc::File *file;

public:
  FileReader(::FILE *init_file) {
    file = reinterpret_cast<__llvm_libc::File *>(init_file);
    file->lock();
  }

  ~FileReader() { file->unlock(); }

  char get_char();
  void unget_char(char c);
};

} // namespace scanf_core
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_STDIO_SCANF_CORE_FILE_READER_H
