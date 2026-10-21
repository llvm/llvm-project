//===-- Implementation of fgetc for baremetal -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/fgetc.h"

#include "hdr/stdio_macros.h" // for EOF.
#include "hdr/types/FILE.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/stdio/baremetal/file_internal.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, fgetc, (::FILE * stream)) {
  unsigned char c;
  auto result = read_internal(reinterpret_cast<char *>(&c), 1, stream);
  if (result.has_error())
    libc_errno = result.error;

  if (result.value != 1)
    return EOF;
  return c;
}

} // namespace LIBC_NAMESPACE_DECL
