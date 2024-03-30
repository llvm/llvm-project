//===-- Linux implementation of pathconf ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/unistd/pread.h"

#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/common.h"
#include "src/__support/macros/sanitizer.h" // for MSAN_UNPOISON
#include "src/errno/libc_errno.h"
#include <stdint.h> // For uint64_t.
#include <sys/fstatvfs.h>
#include <sys/syscall.h> // For syscall numbers.

namespace LIBC_NAMESPACE {

static long pathconfig(const struct statvfs &s, int name) {
  switch (name) {
  case _PC_LINK_MAX:
    return _POSIX_LINK_MAX;

  case _PC_MAX_CANON:
    return _POSIX_MAX_CANON;

  case _PC_MAX_INPUT:
    return _POSIX_MAX_INPUT;

  case _PC_NAME_MAX:
    return _POSIX_NAME_MAX;

  case _PC_PATH_MAX:
    return _POSIX_PATH_MAX;

  case _PC_PIPE_BUF:
    return _POSIX_PIPE_BUF;

  case _PC_CHOWN_RESTRICTED:
    return _POSIX_CHOWN_RESTRICTED;

  case _PC_NO_TRUNC:
    return _POSIX_NO_TRUNC;

  case _PC_VDISABLE:
    return _POSIX_VDISABLE;
  }
}

LLVM_LIBC_FUNCTION(long, pathconf, (char *path, int name)) {
  struct statvfs sb;
  if (fstatvfs(path, &sb) == -1) {
    return -1;
  }
  return pathconfig(sb, name);
}

} // namespace LIBC_NAMESPACE
