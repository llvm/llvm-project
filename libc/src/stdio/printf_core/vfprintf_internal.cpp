//===-- Internal implementation of vfprintf ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/printf_core/vfprintf_internal.h"

#include "src/__support/arg_list.h"
#include "src/stdio/printf_core/file_writer.h"
#include "src/stdio/printf_core/printf_main.h"
#include "src/stdio/printf_core/writer.h"

#include <stdio.h>

namespace __llvm_libc {
namespace printf_core {

int vfprintf_internal(::FILE *__restrict stream, const char *__restrict format,
                      internal::ArgList &args) {
  FileWriter file_writer(stream);
  printf_core::Writer writer(reinterpret_cast<void *>(&file_writer),
                             printf_core::FileWriter::write_str,
                             printf_core::FileWriter::write_chars,
                             printf_core::FileWriter::write_char);
  return printf_core::printf_main(&writer, format, args);
}

} // namespace printf_core
} // namespace __llvm_libc
