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

#include "hdr/fcntl_macros.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include <sys/syscall.h> // For syscall numbers.

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, access, (const char *path, int mode)) {
#ifdef SYS_access
  int ret = LIBC_NAMESPACE::syscall_impl<int>(SYS_access, path, mode);
#elif defined(SYS_faccessat)
  int ret =
      LIBC_NAMESPACE::syscall_impl<int>(SYS_faccessat, AT_FDCWD, path, mode, 0);
#else
#error "access and faccessat syscalls not available."
#endif

  if (ret < 0) {
    libc_errno = -ret;
    return -1;
  }
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
