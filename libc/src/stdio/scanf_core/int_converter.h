//===-- Int type specifier converter for scanf ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_SCANF_CORE_INT_CONVERTER_H
#define LLVM_LIBC_SRC_STDIO_SCANF_CORE_INT_CONVERTER_H

#include "src/stdio/scanf_core/core_structs.h"
#include "src/stdio/scanf_core/reader.h"

#include <stddef.h>

namespace LIBC_NAMESPACE {
namespace scanf_core {

int convert_int(Reader *reader, const FormatSection &to_conv);

} // namespace scanf_core
} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_STDIO_SCANF_CORE_INT_CONVERTER_H
