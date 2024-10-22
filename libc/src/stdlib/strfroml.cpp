//===-- Implementation of strfroml ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/strfroml.h"
#include "src/__support/macros/config.h"
#include "src/stdlib/str_from_util.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, strfroml,
                   (char *__restrict s, size_t n, const char *__restrict format,
                    long double fp)) {
  LIBC_ASSERT(s != nullptr);

  printf_core::FormatSection section =
      internal::parse_format_string(format, fp);

  // To ensure that the conversion function actually uses long double,
  // the length modifier has to be set to LenghtModifier::L
  section.length_modifier = printf_core::LengthModifier::L;

  printf_core::WriteBuffer wb(s, (n > 0 ? n - 1 : 0));
  printf_core::Writer writer(&wb);

  int result = 0;
  if (section.has_conv)
    result = internal::strfromfloat_convert<long double>(&writer, section);
  else
    result = writer.write(section.raw_string);

  if (result < 0)
    return result;

  if (n > 0)
    wb.buff[wb.buff_cur] = '\0';

  return writer.get_chars_written();
}

} // namespace LIBC_NAMESPACE_DECL
