//===-- FILE Writer implementation for printf -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/printf_core/file_writer.h"
#include "src/__support/CPP/string_view.h"
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

int FileWriter::write_str(void *raw_pointer, cpp::string_view new_string) {
  FileWriter *file_writer = reinterpret_cast<FileWriter *>(raw_pointer);
  return file_writer->write(new_string.data(), new_string.size());
}

int FileWriter::write_chars(void *raw_pointer, char new_char, size_t len) {
  FileWriter *file_writer = reinterpret_cast<FileWriter *>(raw_pointer);
  constexpr size_t BUFF_SIZE = 8;
  char buff[BUFF_SIZE] = {new_char};
  int result;
  while (len > BUFF_SIZE) {
    result = file_writer->write(buff, BUFF_SIZE);
    if (result < 0)
      return result;
    len -= BUFF_SIZE;
  }
  return file_writer->write(buff, len);
}

// TODO(michaelrj): Move this to putc_unlocked once that is available.
int FileWriter::write_char(void *raw_pointer, char new_char) {
  FileWriter *file_writer = reinterpret_cast<FileWriter *>(raw_pointer);
  return file_writer->write(&new_char, 1);
}

} // namespace printf_core
} // namespace __llvm_libc
