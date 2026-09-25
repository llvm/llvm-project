//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of tcgetpgrp.
///
//===----------------------------------------------------------------------===//

#include "src/unistd/tcgetpgrp.h"

#include "hdr/sys_ioctl_macros.h"
#include "hdr/types/pid_t.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/ioctl.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(pid_t, tcgetpgrp, (int fd)) {
  pid_t pgid = 0;
  auto result = linux_syscalls::ioctl(fd, TIOCGPGRP, &pgid);
  if (!result.has_value()) {
    libc_errno = result.error();
    return -1;
  }
  return pgid;
}

} // namespace LIBC_NAMESPACE_DECL
