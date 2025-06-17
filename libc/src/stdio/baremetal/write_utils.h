//===-- Baremetal helper functions for writing to stdout/stderr -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/stdio_macros.h" // For stdout/err
#include "hdr/types/FILE.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/OSUtil/io.h"
#include "src/__support/macros/config.h"
#include "src/stdio/printf_core/core_structs.h" // For printf_core::WRITE_OK

namespace LIBC_NAMESPACE_DECL {
namespace write_utils {

LIBC_INLINE int stdout_write_hook(cpp::string_view new_str, void *) {
  write_to_stdout(new_str);
  return printf_core::WRITE_OK;
}

LIBC_INLINE int stderr_write_hook(cpp::string_view new_str, void *) {
  write_to_stderr(new_str);
  return printf_core::WRITE_OK;
}

LIBC_INLINE void write(::FILE *f, cpp::string_view new_str) {
  if (f == stdout) {
    write_to_stdout(new_str);
  } else {
    write_to_stderr(new_str);
  }
}

using StreamWriter = int (*)(cpp::string_view, void *);
LIBC_INLINE StreamWriter get_write_hook(::FILE *f) {
  if (f == stdout)
    return &stdout_write_hook;

  return &stderr_write_hook;
}

} // namespace write_utils
} // namespace LIBC_NAMESPACE_DECL
