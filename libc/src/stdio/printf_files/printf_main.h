//===-- Starting point for printf -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_PRINTF_FILES_PRINTF_MAIN_H
#define LLVM_LIBC_SRC_STDIO_PRINTF_FILES_PRINTF_MAIN_H

#include "src/stdio/printf_files/converter.h"
#include "src/stdio/printf_files/core_structs.h"
#include "src/stdio/printf_files/parser.h"
#include "src/stdio/printf_files/writer.h"

#include <stdarg.h>
#include <stddef.h>

namespace __llvm_libc {
namespace printf_core {

int printf_main(Writer *writer, const char *__restrict str, va_list vlist) {
  Parser parser(str, &vlist);
  Converter converter(writer);

  for (FormatSection cur_section = parser.get_next_section();
       cur_section.raw_len > 0; cur_section = parser.get_next_section()) {
    if (cur_section.has_conv)
      converter.convert(cur_section);
    else
      writer->write(cur_section.raw_string, cur_section.raw_len);
  }

  return writer->get_chars_written();
}

} // namespace printf_core
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_STDIO_PRINTF_FILES_PRINTF_MAIN_H
