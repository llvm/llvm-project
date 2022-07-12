//===-- FILE Writer implementation for printf -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/printf_core/file_writer.h"
#include "src/__support/File/file.h"
#include "src/stdio/printf_core/core_structs.h"
#include <stddef.h>

namespace __llvm_libc {
namespace printf_core {

int FileWriter::write(const char *__restrict to_write, size_t len) {
  int written = file->write_unlocked(to_write, len);
  if (written != static_cast<int>(len))
    written = FILE_WRITE_ERROR;
  if (file->error_unlocked())
    written = FILE_STATUS_ERROR;
  return written;
}

int write_to_file(void *raw_pointer, const char *__restrict to_write,
                  size_t len) {
  FileWriter *file_writer = reinterpret_cast<FileWriter *>(raw_pointer);
  return file_writer->write(to_write, len);
}

} // namespace printf_core
} // namespace __llvm_libc
