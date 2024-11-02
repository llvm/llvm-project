//===-- Linux implementation of tcgetsid ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/termios/tcgetsid.h"

#include "src/__support/OSUtil/syscall.h"
#include "src/__support/common.h"

#include <asm/ioctls.h> // Safe to include without the risk of name pollution.
#include <errno.h>
#include <sys/syscall.h> // For syscall numbers
#include <termios.h>

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(pid_t, tcgetsid, (int fd)) {
  pid_t sid;
  long ret = __llvm_libc::syscall_impl(SYS_ioctl, fd, TIOCGSID, &sid);
  if (ret < 0) {
    errno = -ret;
    return -1;
  }
  return sid;
}

} // namespace __llvm_libc
