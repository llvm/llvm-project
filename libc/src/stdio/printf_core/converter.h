//===-- Format specifier converter for printf -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_PRINTF_CORE_CONVERTER_H
#define LLVM_LIBC_SRC_STDIO_PRINTF_CORE_CONVERTER_H

#include "src/__support/macros/config.h"
#include "src/stdio/printf_core/core_structs.h"
#include "src/stdio/printf_core/writer.h"

#include <stddef.h>

namespace LIBC_NAMESPACE_DECL {
namespace printf_core {

// convert will call a conversion function to convert the FormatSection into
// its string representation, and then that will write the result to the
// writer.
int convert(Writer *writer, const FormatSection &to_conv);

} // namespace printf_core
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_STDIO_PRINTF_CORE_CONVERTER_H
