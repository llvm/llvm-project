//===-- Implementation of getchar for baremetal -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/getchar.h"

#include "hdr/stdio_macros.h" // for EOF.
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/stdio/baremetal/file_internal.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, getchar, ()) {
  unsigned char c;
  auto result = read_internal(reinterpret_cast<char *>(&c), 1, stdin);
  if (result.has_error())
    libc_errno = result.error;

  if (result.value != 1)
    return EOF;
  return c;
}

} // namespace LIBC_NAMESPACE_DECL
