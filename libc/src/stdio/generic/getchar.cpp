//===-- Implementation of getchar -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/getchar.h"
#include "src/__support/File/file.h"

#include "hdr/types/FILE.h"
#include "src/__support/macros/config.h"
#include "src/errno/libc_errno.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, getchar, ()) {
  unsigned char c;
  auto result = stdin->read(&c, 1);
  if (result.has_error())
    libc_errno = result.error;

  if (result.value != 1)
    return EOF;
  return c;
}

} // namespace LIBC_NAMESPACE_DECL
