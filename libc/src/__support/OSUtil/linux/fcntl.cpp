//===-- Implementation of internal fcntl ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/OSUtil/fcntl.h"

#include "hdr/fcntl_macros.h"
#include "hdr/types/struct_f_owner_ex.h"
#include "hdr/types/struct_flock.h"
#include "hdr/types/struct_flock64.h"
#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/common.h"
#include "src/errno/libc_errno.h"

#include <stdarg.h>
#include <sys/syscall.h> // For syscall numbers.

namespace LIBC_NAMESPACE::internal {

int fcntl(int fd, int cmd, void *arg) {
  switch (cmd) {
  case F_OFD_SETLKW: {
    struct flock *flk = reinterpret_cast<struct flock *>(arg);
    // convert the struct to a flock64
    struct flock64 flk64;
    flk64.l_type = flk->l_type;
    flk64.l_whence = flk->l_whence;
    flk64.l_start = flk->l_start;
    flk64.l_len = flk->l_len;
    flk64.l_pid = flk->l_pid;
    // create a syscall
    return LIBC_NAMESPACE::syscall_impl<int>(SYS_fcntl, fd, cmd, &flk64);
  }
  case F_OFD_GETLK:
  case F_OFD_SETLK: {
    struct flock *flk = reinterpret_cast<struct flock *>(arg);
    // convert the struct to a flock64
    struct flock64 flk64;
    flk64.l_type = flk->l_type;
    flk64.l_whence = flk->l_whence;
    flk64.l_start = flk->l_start;
    flk64.l_len = flk->l_len;
    flk64.l_pid = flk->l_pid;
    // create a syscall
    int retVal = LIBC_NAMESPACE::syscall_impl<int>(SYS_fcntl, fd, cmd, &flk64);
    // On failure, return
    if (retVal == -1)
      return -1;
    // Check for overflow, i.e. the offsets are not the same when cast
    // to off_t from off64_t.
    if (static_cast<off_t>(flk64.l_len) != flk64.l_len ||
        static_cast<off_t>(flk64.l_start) != flk64.l_start) {
      libc_errno = EOVERFLOW;
      return -1;
    }
    // Now copy back into flk, in case flk64 got modified
    flk->l_type = flk64.l_type;
    flk->l_whence = flk64.l_whence;
    flk->l_start = static_cast<decltype(flk->l_start)>(flk64.l_start);
    flk->l_len = static_cast<decltype(flk->l_len)>(flk64.l_len);
    flk->l_pid = flk64.l_pid;
    return retVal;
  }
  case F_GETOWN: {
    struct f_owner_ex fex;
    int retVal =
        LIBC_NAMESPACE::syscall_impl<int>(SYS_fcntl, fd, F_GETOWN_EX, &fex);
    if (retVal == -EINVAL)
      return LIBC_NAMESPACE::syscall_impl<int>(SYS_fcntl, fd, cmd,
                                               reinterpret_cast<void *>(arg));
    if (static_cast<unsigned long>(retVal) <= -4096UL)
      return fex.type == F_OWNER_PGRP ? -fex.pid : fex.pid;

    libc_errno = -retVal;
    return -1;
  }
  // The general case
  default: {
    int retVal = LIBC_NAMESPACE::syscall_impl<int>(
        SYS_fcntl, fd, cmd, reinterpret_cast<void *>(arg));
    if (retVal >= 0) {
      return retVal;
    }
    libc_errno = -retVal;
    return -1;
  }
  }
}

} // namespace LIBC_NAMESPACE::internal
