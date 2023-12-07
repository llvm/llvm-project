//===-- Pointer Converter for printf ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_PRINTF_CORE_PTR_CONVERTER_H
#define LLVM_LIBC_SRC_STDIO_PRINTF_CORE_PTR_CONVERTER_H

#include "src/__support/CPP/string_view.h"
#include "src/__support/common.h"
#include "src/stdio/printf_core/converter_utils.h"
#include "src/stdio/printf_core/core_structs.h"
#include "src/stdio/printf_core/int_converter.h"
#include "src/stdio/printf_core/writer.h"

namespace __llvm_libc {
namespace printf_core {

LIBC_INLINE int convert_pointer(Writer *writer, const FormatSection &to_conv) {
  if (to_conv.conv_val_ptr == (void *)(nullptr)) {
    RET_IF_RESULT_NEGATIVE(writer->write("(nullptr)"));
  } else {
    FormatSection hex_conv;
    hex_conv.has_conv = true;
    hex_conv.conv_name = 'x';
    hex_conv.flags = FormatFlags::ALTERNATE_FORM;
    hex_conv.conv_val_raw = reinterpret_cast<uintptr_t>(to_conv.conv_val_ptr);
    return convert_int(writer, hex_conv);
  }
  return WRITE_OK;
}

} // namespace printf_core
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_STDIO_PRINTF_CORE_PTR_CONVERTER_H
