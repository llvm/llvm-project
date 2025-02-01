//===-- Format specifier converter for strftime -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_STRFTIME_CORE_CONVERTER_H
#define LLVM_LIBC_SRC_STDIO_STRFTIME_CORE_CONVERTER_H

#include "src/stdio/printf_core/writer.h"
#include "src/time/strftime_core/core_structs.h"

#include <stddef.h>

namespace LIBC_NAMESPACE_DECL {
namespace strftime_core {

// convert will call a conversion function to convert the FormatSection into
// its string representation, and then that will write the result to the
// writer.
int convert(printf_core::Writer *writer, const FormatSection &to_conv,
            const tm *timeptr);

} // namespace strftime_core
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_STDIO_STRFTIME_CORE_CONVERTER_H
