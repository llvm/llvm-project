//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of tcsetwinsize.
///
//===----------------------------------------------------------------------===//

#include "src/termios/tcsetwinsize.h"

#include "hdr/sys_ioctl_macros.h"
#include "hdr/types/struct_winsize.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/ioctl.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, tcsetwinsize, (int fd, const struct winsize *ws)) {
  auto ret = linux_syscalls::ioctl(fd, TIOCSWINSZ, ws);
  if (!ret.has_value()) {
    libc_errno = ret.error();
    return -1;
  }
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
