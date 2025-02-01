//===-- Starting point for strftime ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/time/strftime_core/strftime_main.h"

#include "src/stdio/printf_core/writer.h"
#include "src/time/strftime_core/converter.h"
#include "src/time/strftime_core/core_structs.h"
#include "src/time/strftime_core/parser.h"

#include "hdr/types/struct_tm.h"

namespace LIBC_NAMESPACE_DECL {
namespace strftime_core {

int strftime_main(printf_core::Writer *writer, const char *__restrict str,
                  const struct tm *timeptr) {
  Parser parser(str);
  int result = 0;
  for (FormatSection cur_section = parser.get_next_section();
       !cur_section.raw_string.empty();
       cur_section = parser.get_next_section()) {
    if (cur_section.has_conv)
      result = convert(writer, cur_section, timeptr);
    else
      result = writer->write(cur_section.raw_string);

    if (result < 0)
      return result;
  }

  return writer->get_chars_written();
}

} // namespace strftime_core
} // namespace LIBC_NAMESPACE_DECL
