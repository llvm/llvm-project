//===-- Format specifier converter implmentation for printf -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/printf_core/converter.h"

#include "src/stdio/printf_core/core_structs.h"
#include "src/stdio/printf_core/writer.h"

// This option allows for replacing all of the conversion functions with custom
// replacements. This allows conversions to be replaced at compile time.
#ifndef LLVM_LIBC_PRINTF_CONV_ATLAS
#include "src/stdio/printf_core/converter_atlas.h"
#else
#include LLVM_LIBC_PRINTF_CONV_ATLAS
#endif

#include <stddef.h>

namespace __llvm_libc {
namespace printf_core {

void convert(Writer *writer, FormatSection to_conv) {
  if (!to_conv.has_conv) {
    writer->write(to_conv.raw_string, to_conv.raw_len);
    return;
  }
  switch (to_conv.conv_name) {
  case '%':
    writer->write("%", 1);
    return;
  case 'c':
    convert_char(writer, to_conv);
    return;
  case 's':
    convert_string(writer, to_conv);
    return;
  case 'd':
  case 'i':
  case 'u':
    // convert_int(writer, to_conv);
    return;
  case 'o':
    // convert_oct(writer, to_conv);
    return;
  case 'x':
  case 'X':
    // convert_hex(writer, to_conv);
    return;
  // TODO(michaelrj): add a flag to disable float point values here
  case 'f':
  case 'F':
    // convert_float_decimal(writer, to_conv);
    return;
  case 'e':
  case 'E':
    // convert_float_dec_exp(writer, to_conv);
    return;
  case 'a':
  case 'A':
    // convert_float_hex_exp(writer, to_conv);
    return;
  case 'g':
  case 'G':
    // convert_float_mixed(writer, to_conv);
    return;
  // TODO(michaelrj): add a flag to disable writing an int here
  case 'n':
    // convert_write_int(writer, to_conv);
    return;
  case 'p':
    // convert_pointer(writer, to_conv);
    return;
  default:
    writer->write(to_conv.raw_string, to_conv.raw_len);
    return;
  }
}

} // namespace printf_core
} // namespace __llvm_libc
