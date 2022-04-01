//===-- Format specifier converter for printf -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_PRINTF_FILES_CONVERTER_H
#define LLVM_LIBC_SRC_STDIO_PRINTF_FILES_CONVERTER_H

#include "src/stdio/printf_files/core_structs.h"
#include "src/stdio/printf_files/writer.h"

#include <stddef.h>

namespace __llvm_libc {
namespace printf_core {

class Converter {
  Writer *writer;

public:
  Converter(Writer *writer);

  // convert will call a conversion function to convert the FormatSection into
  // its string representation, and then that will write the result to the
  // writer.
  void convert(FormatSection to_conv);
};

} // namespace printf_core
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_STDIO_PRINTF_FILES_CONVERTER_H
