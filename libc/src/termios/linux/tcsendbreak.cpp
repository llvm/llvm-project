//===-- Linux implementation of tcsendbreak -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/termios/tcsendbreak.h"

#include "src/__support/OSUtil/syscall.h"
#include "src/__support/common.h"

#include <asm/ioctls.h> // Safe to include without the risk of name pollution.
#include <errno.h>
#include <sys/syscall.h> // For syscall numbers
#include <termios.h>

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(pid_t, tcsendbreak, (int fd, int /* unused duration */)) {
  // POSIX leaves the behavior for non-zero duration implementation dependent.
  // Which means that the behavior can be the same as it is when duration is
  // zero. So, we just pass zero to the syscall.
  long ret = __llvm_libc::syscall_impl(SYS_ioctl, fd, TCSBRK, 0);
  if (ret < 0) {
    errno = -ret;
    return -1;
  }
  return 0;
}

} // namespace __llvm_libc
