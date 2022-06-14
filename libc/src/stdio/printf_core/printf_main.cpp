//===-- Starting point for printf -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/printf_core/printf_main.h"

#include "src/__support/arg_list.h"
#include "src/stdio/printf_core/converter.h"
#include "src/stdio/printf_core/core_structs.h"
#include "src/stdio/printf_core/parser.h"
#include "src/stdio/printf_core/writer.h"

#include <stddef.h>

namespace __llvm_libc {
namespace printf_core {

int printf_main(Writer *writer, const char *__restrict str,
                internal::ArgList &args) {
  Parser parser(str, args);
  int result = 0;
  for (FormatSection cur_section = parser.get_next_section();
       cur_section.raw_len > 0; cur_section = parser.get_next_section()) {
    if (cur_section.has_conv)
      result = convert(writer, cur_section);
    else
      result = writer->write(cur_section.raw_string, cur_section.raw_len);

    if (result < 0)
      return result;
  }

  return writer->get_chars_written();
}

} // namespace printf_core
} // namespace __llvm_libc
