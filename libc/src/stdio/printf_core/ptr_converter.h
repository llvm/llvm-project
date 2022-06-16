//===-- Pointer Converter for printf ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_PRINTF_CORE_PTR_CONVERTER_H
#define LLVM_LIBC_SRC_STDIO_PRINTF_CORE_PTR_CONVERTER_H

#include "src/stdio/printf_core/converter_utils.h"
#include "src/stdio/printf_core/core_structs.h"
#include "src/stdio/printf_core/hex_converter.h"
#include "src/stdio/printf_core/writer.h"

namespace __llvm_libc {
namespace printf_core {

int inline convert_pointer(Writer *writer, const FormatSection &to_conv) {
  if (to_conv.conv_val_ptr == (void *)(nullptr)) {
    const char ZERO_STR[] = "(nullptr)";
    // subtract 1 from sizeof to remove the null byte at the end.
    RET_IF_RESULT_NEGATIVE(writer->write(ZERO_STR, sizeof(ZERO_STR) - 1));
  } else {
    FormatSection hex_conv;
    hex_conv.has_conv = true;
    hex_conv.conv_name = 'x';
    hex_conv.flags = FormatFlags::ALTERNATE_FORM;
    hex_conv.conv_val_raw = reinterpret_cast<uintptr_t>(to_conv.conv_val_ptr);
    return convert_hex(writer, hex_conv);
  }
  return 0;
}

} // namespace printf_core
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_STDIO_PRINTF_CORE_PTR_CONVERTER_H
