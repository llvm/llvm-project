//===-- Linux implementation of unlink ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/unistd/unlink.h"

#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/common.h"

#include <errno.h>
#include <fcntl.h>
#include <sys/syscall.h> // For syscall numbers.

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(int, unlink, (const char *path)) {
#ifdef SYS_unlink
  long ret = __llvm_libc::syscall_impl(SYS_unlink, path);
#elif defined(SYS_unlinkat)
  long ret = __llvm_libc::syscall_impl(SYS_unlinkat, AT_FDCWD, path, 0);
#else
#error "Unlink syscalls not available."
#endif

  if (ret < 0) {
    errno = -ret;
    return -1;
  }
  return 0;
}

} // namespace __llvm_libc
