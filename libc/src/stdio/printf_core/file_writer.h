//===-- FILE Writer class for printf ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_PRINTF_CORE_FILE_WRITER_H
#define LLVM_LIBC_SRC_STDIO_PRINTF_CORE_FILE_WRITER_H

#include "src/__support/File/file.h"
#include <stddef.h>

namespace __llvm_libc {
namespace printf_core {

// write_to_file treats raw_pointer as a File and calls its write
// function.
void write_to_file(void *raw_pointer, const char *__restrict to_write,
                   size_t len) {
  __llvm_libc::File *file = reinterpret_cast<__llvm_libc::File *>(raw_pointer);
  file->write(to_write, len);
}

} // namespace printf_core
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_STDIO_PRINTF_CORE_FILE_WRITER_H
