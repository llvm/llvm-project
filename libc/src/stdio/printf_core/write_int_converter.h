//===-- Write integer Converter for printf ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_PRINTF_CORE_WRITE_INT_CONVERTER_H
#define LLVM_LIBC_SRC_STDIO_PRINTF_CORE_WRITE_INT_CONVERTER_H

#include "src/__support/CPP/limits.h"
#include "src/stdio/printf_core/core_structs.h"
#include "src/stdio/printf_core/writer.h"

#include <inttypes.h>
#include <stddef.h>

namespace __llvm_libc {
namespace printf_core {

int inline convert_write_int(Writer *writer, const FormatSection &to_conv) {

  // This is an additional check added by LLVM-libc. The reason it returns -3 is
  // because printf uses negative return values for errors, and -1 and -2 are
  // already in use by the file_writer class for file errors.
  if (to_conv.conv_val_ptr == nullptr)
    return NULLPTR_WRITE_ERROR;

  int written = writer->get_chars_written();

  switch (to_conv.length_modifier) {
  case LengthModifier::none:
    *reinterpret_cast<int *>(to_conv.conv_val_ptr) = written;
    break;
  case LengthModifier::l:
    *reinterpret_cast<long *>(to_conv.conv_val_ptr) = written;
    break;
  case LengthModifier::ll:
  case LengthModifier::L:
    *reinterpret_cast<long long *>(to_conv.conv_val_ptr) = written;
    break;
  case LengthModifier::h:
    *reinterpret_cast<short *>(to_conv.conv_val_ptr) = written;
    break;
  case LengthModifier::hh:
    *reinterpret_cast<signed char *>(to_conv.conv_val_ptr) = written;
    break;
  case LengthModifier::z:
    *reinterpret_cast<size_t *>(to_conv.conv_val_ptr) = written;
    break;
  case LengthModifier::t:
    *reinterpret_cast<ptrdiff_t *>(to_conv.conv_val_ptr) = written;
    break;
  case LengthModifier::j:
    *reinterpret_cast<uintmax_t *>(to_conv.conv_val_ptr) = written;
    break;
  }
  return WRITE_OK;
}

} // namespace printf_core
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_STDIO_PRINTF_CORE_WRITE_INT_CONVERTER_H
