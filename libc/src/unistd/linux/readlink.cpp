//===-- Linux implementation of readlink ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/unistd/readlink.h"

#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/common.h"

#include <errno.h>
#include <fcntl.h>
#include <sys/syscall.h> // For syscall numbers.

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(ssize_t, readlink,
                   (const char *__restrict path, char *__restrict buf,
                    size_t bufsize)) {
#ifdef SYS_readlink
  ssize_t ret = __llvm_libc::syscall_impl(SYS_readlink, path, buf, bufsize);
#elif defined(SYS_readlinkat)
  ssize_t ret =
      __llvm_libc::syscall_impl(SYS_readlinkat, AT_FDCWD, path, buf, bufsize);
#else
#error "SYS_readlink or SYS_readlinkat not available."
#endif
  if (ret < 0) {
    errno = -ret;
    return -1;
  }
  return ret;
}

} // namespace __llvm_libc
