//===-- Linux implementation of remove ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/remove.h"

#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/common.h"

#include <errno.h>
#include <fcntl.h>       // For AT_* macros.
#include <sys/syscall.h> // For syscall numbers.

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(int, remove, (const char *path)) {
  // We first try unlinking it as a file. If it is ia file, it will succeed. If
  // it fails with EISDIR, we will try unlinking it as a directory.
  int ret = __llvm_libc::syscall(SYS_unlinkat, AT_FDCWD, path, 0);
  if (ret == -EISDIR)
    ret = __llvm_libc::syscall(SYS_unlinkat, AT_FDCWD, path, AT_REMOVEDIR);
  if (ret >= 0)
    return 0;
  errno = -ret;
  return -1;
}

} // namespace __llvm_libc
