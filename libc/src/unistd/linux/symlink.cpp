//===-- Linux implementation of symlink -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/unistd/symlink.h"

#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/common.h"

#include <errno.h>
#include <fcntl.h>
#include <sys/syscall.h> // For syscall numbers.

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(int, symlink, (const char *path1, const char *path2)) {
#ifdef SYS_symlink
  long ret = __llvm_libc::syscall(SYS_symlink, path1, path2);
#elif defined(SYS_symlinkat)
  long ret = __llvm_libc::syscall(SYS_symlinkat, path1, AT_FDCWD, path2);
#else
#error "SYS_symlink or SYS_symlinkat not available."
#endif
  if (ret < 0) {
    errno = -ret;
    return -1;
  }
  return ret;
}

} // namespace __llvm_libc
