//===-- Implementation of strfromf ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/strfromf.h"
#include "src/stdlib/str_from_util.h"

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(int, strfromf,
                   (char *__restrict s, size_t n, const char *__restrict format,
                    float fp)) {
  LIBC_ASSERT(s != nullptr);

  printf_core::FormatSection section =
      internal::parse_format_string(format, fp);
  printf_core::WriteBuffer wb(s, (n > 0 ? n - 1 : 0));
  printf_core::Writer writer(&wb);

  int result = 0;
  if (section.has_conv)
    result = internal::strfromfloat_convert<float>(&writer, section);
  else
    result = writer.write(section.raw_string);

  if (result < 0)
    return result;

  if (n > 0)
    wb.buff[wb.buff_cur] = '\0';

  return writer.get_chars_written();
}

} // namespace LIBC_NAMESPACE
