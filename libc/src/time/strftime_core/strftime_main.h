//===-- Starting point for strftime -------------------------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_STRFTIME_CORE_STRFTIME_MAIN_H
#define LLVM_LIBC_SRC_STDIO_STRFTIME_CORE_STRFTIME_MAIN_H

#include "src/__support/macros/config.h"
#include "src/stdio/printf_core/writer.h"

#include <stddef.h>
#include <time.h>

namespace LIBC_NAMESPACE_DECL {
namespace strftime_core {

// Passed to writeBuffer so error is returned if there is not enough buffer
// space.
LIBC_INLINE int overflow_write_mock(cpp::string_view, void *) { return -1; }

int strftime_main(printf_core::Writer *writer, const char *__restrict str,
                  const struct tm *timeptr);

} // namespace strftime_core
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_STDIO_STRFTIME_CORE_STRFTIME_MAIN_H
