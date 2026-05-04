//===-- Linux implementation of faccessat ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/unistd/faccessat.h"

#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/common.h"

#include "hdr/fcntl_macros.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include <sys/syscall.h> // For syscall numbers.

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, faccessat,
                   (int fd, const char *path, int amode, int flag)) {
#ifdef SYS_faccessat2
  int ret =
      LIBC_NAMESPACE::syscall_impl<int>(SYS_faccessat2, fd, path, amode, flag);
#else
#error "faccessat2 syscall is not available."
#endif

  if (ret < 0) {
    libc_errno = -ret;
    return -1;
  }
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
