//===-- Linux implementation of tcdrain -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/termios/tcdrain.h"

#include "src/__support/OSUtil/syscall.h"
#include "src/__support/common.h"

#include <asm/ioctls.h> // Safe to include without the risk of name pollution.
#include <errno.h>
#include <sys/syscall.h> // For syscall numbers
#include <termios.h>

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(int, tcdrain, (int fd)) {
  long ret = __llvm_libc::syscall_impl(SYS_ioctl, fd, TCSBRK, 1);
  if (ret < 0) {
    errno = -ret;
    return -1;
  }
  return 0;
}

} // namespace __llvm_libc
