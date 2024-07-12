//===-- Format specifier converter for scanf -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_SCANF_CORE_CONVERTER_H
#define LLVM_LIBC_SRC_STDIO_SCANF_CORE_CONVERTER_H

#include "src/__support/CPP/string_view.h"
#include "src/stdio/scanf_core/core_structs.h"
#include "src/stdio/scanf_core/reader.h"

#include <stddef.h>

namespace LIBC_NAMESPACE {
namespace scanf_core {

// convert will call a conversion function to convert the FormatSection into
// its string representation, and then that will write the result to the
// reader.
int convert(Reader *reader, const FormatSection &to_conv);

// raw_match takes a raw string and matches it to the characters obtained from
// the reader.
int raw_match(Reader *reader, cpp::string_view raw_string);

} // namespace scanf_core
} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_STDIO_SCANF_CORE_CONVERTER_H
