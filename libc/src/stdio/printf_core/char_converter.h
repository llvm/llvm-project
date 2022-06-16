//===-- String Converter for printf -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_PRINTF_CORE_CHAR_CONVERTER_H
#define LLVM_LIBC_SRC_STDIO_PRINTF_CORE_CHAR_CONVERTER_H

#include "src/stdio/printf_core/core_structs.h"
#include "src/stdio/printf_core/writer.h"

namespace __llvm_libc {
namespace printf_core {

int inline convert_char(Writer *writer, const FormatSection &to_conv) {
  char c = to_conv.conv_val_raw;

  if (to_conv.min_width > 1) {
    if ((to_conv.flags & FormatFlags::LEFT_JUSTIFIED) ==
        FormatFlags::LEFT_JUSTIFIED) {
      RET_IF_RESULT_NEGATIVE(writer->write(&c, 1));
      RET_IF_RESULT_NEGATIVE(writer->write_chars(' ', to_conv.min_width - 1));
    } else {
      RET_IF_RESULT_NEGATIVE(writer->write_chars(' ', to_conv.min_width - 1));
      RET_IF_RESULT_NEGATIVE(writer->write(&c, 1));
    }
  } else {
    RET_IF_RESULT_NEGATIVE(writer->write(&c, 1));
  }
  return 0;
}

} // namespace printf_core
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_STDIO_PRINTF_CORE_CHAR_CONVERTER_H
