//===-- Linux implementation of rename ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/rename.h"
#include "hdr/fcntl_macros.h"
#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/errno/libc_errno.h"
#include <sys/syscall.h> // For syscall numbers.

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, rename, (const char *oldpath, const char *newpath)) {
  int ret = LIBC_NAMESPACE::syscall_impl<int>(SYS_renameat2, AT_FDCWD, oldpath,
                                              AT_FDCWD, newpath, 0);

  if (ret >= 0)
    return 0;
  libc_errno = -ret;
  return -1;
}

} // namespace LIBC_NAMESPACE_DECL
