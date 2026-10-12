//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of lockf.
///
//===----------------------------------------------------------------------===//

#include "src/unistd/lockf.h"

#include "hdr/errno_macros.h"
#include "hdr/fcntl_macros.h"
#include "hdr/types/off_t.h"
#include "hdr/types/struct_flock.h"
#include "hdr/unistd_macros.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/fcntl.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/getpid.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, lockf, (int fd, int cmd, off_t len)) {
  struct flock fl = {};
  fl.l_type = F_WRLCK;
  fl.l_whence = SEEK_CUR;
  fl.l_start = 0;
  fl.l_len = len;
  fl.l_pid = 0;

  switch (cmd) {
  case F_TEST: {
    fl.l_type = F_RDLCK;
    auto result = linux_syscalls::fcntl(fd, F_GETLK, &fl);
    if (!result.has_value()) {
      libc_errno = result.error();
      return -1;
    }
    if (fl.l_type == F_UNLCK || fl.l_pid == linux_syscalls::getpid())
      return 0;
    libc_errno = EACCES;
    return -1;
  }
  case F_ULOCK: {
    fl.l_type = F_UNLCK;
    auto result = linux_syscalls::fcntl(fd, F_SETLK, &fl);
    if (!result.has_value()) {
      libc_errno = result.error();
      return -1;
    }
    return 0;
  }
  case F_LOCK: {
    auto result = linux_syscalls::fcntl(fd, F_SETLKW, &fl);
    if (!result.has_value()) {
      libc_errno = result.error();
      return -1;
    }
    return 0;
  }
  case F_TLOCK: {
    auto result = linux_syscalls::fcntl(fd, F_SETLK, &fl);
    if (!result.has_value()) {
      libc_errno = result.error();
      return -1;
    }
    return 0;
  }
  default:
    libc_errno = EINVAL;
    return -1;
  }
}

} // namespace LIBC_NAMESPACE_DECL
