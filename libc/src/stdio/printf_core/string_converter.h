//===-- String Converter for printf -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_PRINTF_CORE_STRING_CONVERTER_H
#define LLVM_LIBC_SRC_STDIO_PRINTF_CORE_STRING_CONVERTER_H

#include "src/stdio/printf_core/converter_utils.h"
#include "src/stdio/printf_core/core_structs.h"
#include "src/stdio/printf_core/writer.h"

#include <stddef.h>

namespace __llvm_libc {
namespace printf_core {

int inline convert_string(Writer *writer, const FormatSection &to_conv) {
  int string_len = 0;

  for (char *cur_str = reinterpret_cast<char *>(to_conv.conv_val_ptr);
       cur_str[string_len]; ++string_len) {
    ;
  }

  if (to_conv.precision >= 0 && to_conv.precision < string_len)
    string_len = to_conv.precision;

  if (to_conv.min_width > string_len) {
    if ((to_conv.flags & FormatFlags::LEFT_JUSTIFIED) ==
        FormatFlags::LEFT_JUSTIFIED) {
      RET_IF_RESULT_NEGATIVE(writer->write(
          reinterpret_cast<const char *>(to_conv.conv_val_ptr), string_len));
      RET_IF_RESULT_NEGATIVE(
          writer->write_chars(' ', to_conv.min_width - string_len));

    } else {
      RET_IF_RESULT_NEGATIVE(
          writer->write_chars(' ', to_conv.min_width - string_len));
      RET_IF_RESULT_NEGATIVE(writer->write(
          reinterpret_cast<const char *>(to_conv.conv_val_ptr), string_len));
    }
  } else {
    RET_IF_RESULT_NEGATIVE(writer->write(
        reinterpret_cast<const char *>(to_conv.conv_val_ptr), string_len));
  }
  return WRITE_OK;
}

} // namespace printf_core
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_STDIO_PRINTF_CORE_STRING_CONVERTER_H
