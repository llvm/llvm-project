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
#include "src/__support/error_or.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

static ErrorOr<int> lockf_impl(int fd, int cmd, off_t len) {
  struct flock fl = {};
  fl.l_whence = SEEK_CUR;
  fl.l_start = 0;
  fl.l_len = len;

  switch (cmd) {
  case F_TEST: {
    fl.l_type = F_RDLCK;
    auto result = linux_syscalls::fcntl(fd, F_GETLK, &fl);
    if (!result.has_value())
      return Error(result.error());
    if (fl.l_type == F_UNLCK || fl.l_pid == linux_syscalls::getpid())
      return 0;
    return Error(EACCES);
  }
  case F_ULOCK:
    fl.l_type = F_UNLCK;
    return linux_syscalls::fcntl(fd, F_SETLK, &fl);
  case F_LOCK:
    fl.l_type = F_WRLCK;
    return linux_syscalls::fcntl(fd, F_SETLKW, &fl);
  case F_TLOCK:
    fl.l_type = F_WRLCK;
    return linux_syscalls::fcntl(fd, F_SETLK, &fl);
  default:
    return Error(EINVAL);
  }
}

LLVM_LIBC_FUNCTION(int, lockf, (int fd, int cmd, off_t len)) {
  auto result = lockf_impl(fd, cmd, len);
  if (!result.has_value()) {
    libc_errno = result.error();
    return -1;
  }
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
