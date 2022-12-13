//===-- Starting point for scanf --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/scanf_core/scanf_main.h"

#include "src/__support/arg_list.h"
#include "src/stdio/scanf_core/converter.h"
#include "src/stdio/scanf_core/core_structs.h"
#include "src/stdio/scanf_core/parser.h"
#include "src/stdio/scanf_core/reader.h"

#include <stddef.h>

namespace __llvm_libc {
namespace scanf_core {

int scanf_main(Reader *reader, const char *__restrict str,
               internal::ArgList &args) {
  Parser parser(str, args);
  int ret_val = READ_OK;
  int conversions = 0;
  for (FormatSection cur_section = parser.get_next_section();
       !cur_section.raw_string.empty() && ret_val == READ_OK;
       cur_section = parser.get_next_section()) {
    if (cur_section.has_conv) {
      ret_val = convert(reader, cur_section);
      conversions += ret_val == READ_OK ? 1 : 0;
    } else {
      ret_val = raw_match(reader, cur_section.raw_string);
    }
  }

  if (conversions == 0 && reader->has_error()) {
    // This is intended to be converted to EOF in the client call to avoid
    // including stdio.h in this internal file.
    return -1;
  }
  return conversions;
}

} // namespace scanf_core
} // namespace __llvm_libc
