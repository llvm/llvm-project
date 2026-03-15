//===-- Implementation of open --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/fcntl/open.h"

#include "hdr/fcntl_macros.h"
#include "hdr/types/mode_t.h"
#include "src/__support/OSUtil/fcntl.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include <stdarg.h>

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, open, (const char *path, int flags, ...)) {
  mode_t mode_flags = 0;
  // O_TMPFILE is a multi-bit flag so we test all bits by checking for equality
  // with O_TMPFILE
  if ((flags & O_CREAT) || (flags & O_TMPFILE) == O_TMPFILE) {
    va_list varargs;
    va_start(varargs, flags);
    mode_flags = va_arg(varargs, mode_t);
    va_end(varargs);
  }

  auto result = internal::open(path, flags, mode_flags);

  if (!result.has_value()) {
    libc_errno = result.error();
    return -1;
  }
  return result.value();
}

} // namespace LIBC_NAMESPACE_DECL
