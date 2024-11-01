//===-- FILE Reader implementation for scanf --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/scanf_core/file_reader.h"
#include "src/__support/File/file.h"
#include <stddef.h>

namespace __llvm_libc {
namespace scanf_core {

char FileReader::get_char() {
  char tiny_buff = 0;
  auto result = file->read_unlocked(&tiny_buff, 1);
  if (result.value != 1 || result.has_error())
    return 0;
  return tiny_buff;
}

void FileReader::unget_char(char c) { file->ungetc_unlocked(c); }

} // namespace scanf_core
} // namespace __llvm_libc
