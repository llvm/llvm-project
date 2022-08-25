//===-- Linux implementation of link --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/unistd/link.h"

#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/common.h"

#include <errno.h>
#include <fcntl.h>
#include <sys/syscall.h> // For syscall numbers.

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(int, link, (const char *path1, const char *path2)) {
#ifdef SYS_link
  long ret = __llvm_libc::syscall(SYS_link, path1, path2);
#elif defined(SYS_linkat)
  long ret =
      __llvm_libc::syscall(SYS_linkat, AT_FDCWD, path1, AT_FDCWD, path2, 0);
#else
#error "SYS_link or SYS_linkat not available."
#endif
  if (ret < 0) {
    errno = -ret;
    return -1;
  }
  return ret;
}

} // namespace __llvm_libc
