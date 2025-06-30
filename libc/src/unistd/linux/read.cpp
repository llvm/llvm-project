//===-- Linux implementation of read --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/unistd/read.h"

#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/sanitizer.h" // for MSAN_UNPOISON
#include <sys/syscall.h>                    // For syscall numbers.

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(ssize_t, read, (int fd, void *buf, size_t count)) {
  ssize_t ret = LIBC_NAMESPACE::syscall_impl<ssize_t>(SYS_read, fd, buf, count);
  if (ret < 0) {
    libc_errno = static_cast<int>(-ret);
    return -1;
  }
  // The cast is important since there is a check that dereferences the pointer
  // which fails on void*.
  MSAN_UNPOISON(reinterpret_cast<char *>(buf), count);
  return ret;
}

} // namespace LIBC_NAMESPACE_DECL
