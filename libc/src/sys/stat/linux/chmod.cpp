//===-- Linux implementation of chmod -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/stat/chmod.h"

#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/common.h"

#include "hdr/fcntl_macros.h"
#include "hdr/types/mode_t.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include <sys/stat.h>
#include <sys/syscall.h> // For syscall numbers.

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, chmod, (const char *path, mode_t mode)) {
#ifdef SYS_chmod
  int ret = LIBC_NAMESPACE::syscall_impl<int>(SYS_chmod, path, mode);
#elif defined(SYS_fchmodat)
  int ret =
      LIBC_NAMESPACE::syscall_impl<int>(SYS_fchmodat, AT_FDCWD, path, mode, 0);
#elif defined(SYS_fchmodat2)
  int ret = LIBC_NAMESPACE::syscall_impl<int>(SYS_fchmodat2, AT_FDCWD, path,
                                              mode, 0, AT_SYMLINK_NOFOLLOW);
#else
#error "chmod, fchmodat and fchmodat2 syscalls not available."
#endif

  if (ret < 0) {
    libc_errno = -ret;
    return -1;
  }
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
