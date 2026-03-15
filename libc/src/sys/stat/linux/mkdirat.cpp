//===-- Linux implementation of mkdirat -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/stat/mkdirat.h"

#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/common.h"

#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include <sys/stat.h>
#include <sys/syscall.h> // For syscall numbers.

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, mkdirat, (int dfd, const char *path, mode_t mode)) {
#ifdef SYS_mkdirat
  int ret = LIBC_NAMESPACE::syscall_impl<int>(SYS_mkdirat, dfd, path, mode);
#else
#error "mkdirat syscall not available."
#endif

  if (ret < 0) {
    libc_errno = -ret;
    return -1;
  }
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
