//===-- Implementation file for writev ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "src/sys/uio/writev.h"
#include "hdr/types/ssize_t.h"
#include "hdr/types/struct_iovec.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/writev.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(ssize_t, writev, (int fd, const iovec *iov, int iovcnt)) {
  auto result = linux_syscalls::writev(fd, iov, iovcnt);
  if (!result) {
    libc_errno = result.error();
    return -1;
  }
  return result.value();
}

} // namespace LIBC_NAMESPACE_DECL
