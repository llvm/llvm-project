//===-- Internal implementation header of vfprintf --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_PRINTF_CORE_VFPRINTF_INTERNAL_H
#define LLVM_LIBC_SRC_STDIO_PRINTF_CORE_VFPRINTF_INTERNAL_H

#include "src/__support/File/file.h"
#include "src/__support/arg_list.h"
#include "src/__support/macros/attributes.h" // For LIBC_INLINE
#include "src/stdio/printf_core/file_writer.h"
#include "src/stdio/printf_core/printf_main.h"
#include "src/stdio/printf_core/writer.h"

#include <stdio.h>

namespace __llvm_libc {
namespace printf_core {

template <typename file_t>
LIBC_INLINE int vfprintf_internal(file_t *__restrict stream,
                                  const char *__restrict format,
                                  internal::ArgList &args) {
  FileWriter<file_t> file_writer(stream);
  Writer writer(reinterpret_cast<void *>(&file_writer),
                FileWriter<file_t>::write_str, FileWriter<file_t>::write_chars,
                FileWriter<file_t>::write_char);
  return printf_main(&writer, format, args);
}

} // namespace printf_core
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_STDIO_PRINTF_CORE_VFPRINTF_INTERNAL_H
