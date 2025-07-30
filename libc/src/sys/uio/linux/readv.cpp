//===-- Implementation file for readv -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "src/sys/uio/readv.h"
#include "hdr/types/ssize_t.h"
#include "hdr/types/struct_iovec.h"
#include "src/__support/OSUtil/syscall.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include <sys/syscall.h>

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(ssize_t, readv, (int fd, const iovec *iov, int iovcnt)) {
  long ret = LIBC_NAMESPACE::syscall_impl<long>(SYS_readv, fd, iov, iovcnt);
  // On failure, return -1 and set errno.
  if (ret < 0) {
    libc_errno = static_cast<int>(-ret);
    return -1;
  }
  // On success, return number of bytes read.
  return static_cast<ssize_t>(ret);
}

} // namespace LIBC_NAMESPACE_DECL
