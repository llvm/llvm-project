//===-- Linux implementation of access ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/unistd/access.h"

#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/common.h"

#include <errno.h>
#include <fcntl.h>
#include <sys/syscall.h> // For syscall numbers.

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(int, access, (const char *path, int mode)) {
#ifdef SYS_access
  long ret = __llvm_libc::syscall(SYS_access, path, mode);
#elif defined(SYS_accessat)
  long ret = __llvm_libc::syscall(SYS_accessat, AT_FDCWD, path, mode);
#else
#error "access syscalls not available."
#endif

  if (ret < 0) {
    errno = -ret;
    return -1;
  }
  return 0;
}

} // namespace __llvm_libc
