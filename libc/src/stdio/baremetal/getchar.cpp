//===-- Baremetal implementation of getchar -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/getchar.h"
#include "src/__support/OSUtil/io.h"
#include "src/__support/macros/config.h"

#include "hdr/stdio_macros.h" // for EOF.

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, getchar, ()) {
  char buf[1];
  auto result = read_from_stdin(buf, sizeof(buf));
  if (result < 0)
    return EOF;
  return buf[0];
}

} // namespace LIBC_NAMESPACE_DECL
