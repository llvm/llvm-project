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
#include "src/__support/macros/attributes.h" // For LIBC_INLINE
#include "src/stdio/printf_core/core_structs.h"

#include <stddef.h>
#include <stdio.h>

namespace __llvm_libc {
namespace printf_core {

template <typename file_t> class FileWriter {
  file_t *file;

public:
  LIBC_INLINE FileWriter(file_t *init_file);

  LIBC_INLINE ~FileWriter();

  LIBC_INLINE int write(const char *__restrict to_write, size_t len);

  // These write functions take a FileWriter as a void* in raw_pointer, and
  // call the appropriate write function on it.
  static int write_str(void *raw_pointer, cpp::string_view new_string) {
    FileWriter *file_writer = reinterpret_cast<FileWriter *>(raw_pointer);
    return file_writer->write(new_string.data(), new_string.size());
  }
  static int write_chars(void *raw_pointer, char new_char, size_t len) {
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
  static int write_char(void *raw_pointer, char new_char) {
    FileWriter *file_writer = reinterpret_cast<FileWriter *>(raw_pointer);
    return file_writer->write(&new_char, 1);
  }
};

// The interface for using our internal file implementation.
template <>
LIBC_INLINE
FileWriter<__llvm_libc::File>::FileWriter(__llvm_libc::File *init_file) {
  file = init_file;
  file->lock();
}
template <> LIBC_INLINE FileWriter<__llvm_libc::File>::~FileWriter() {
  file->unlock();
}
template <>
LIBC_INLINE int
FileWriter<__llvm_libc::File>::write(const char *__restrict to_write,
                                     size_t len) {
  auto result = file->write_unlocked(to_write, len);
  size_t written = result.value;
  if (written != len || result.has_error())
    written = FILE_WRITE_ERROR;
  if (file->error_unlocked())
    written = FILE_STATUS_ERROR;
  return written;
}

// The interface for using the system's file implementation.
template <> LIBC_INLINE FileWriter<::FILE>::FileWriter(::FILE *init_file) {
  file = init_file;
  ::flockfile(file);
}
template <> LIBC_INLINE FileWriter<::FILE>::~FileWriter() {
  ::funlockfile(file);
}
template <>
LIBC_INLINE int FileWriter<::FILE>::write(const char *__restrict to_write,
                                          size_t len) {
  size_t written = ::fwrite_unlocked(to_write, 1, len, file);
  if (written != len || ::ferror_unlocked(file))
    written = FILE_WRITE_ERROR;
  return written;
}

} // namespace printf_core
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_STDIO_PRINTF_CORE_FILE_WRITER_H
