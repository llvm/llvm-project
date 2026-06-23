//===-- Linux implementation of tcflow -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/termios/tcflow.h"

#include "src/__support/OSUtil/syscall.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"

#include <asm/ioctls.h> // Safe to include without the risk of name pollution.
#include <sys/syscall.h> // For syscall numbers
#include <termios.h>

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, tcflow, (int fd, int action)) {
  int ret = LIBC_NAMESPACE::syscall_impl<int>(SYS_ioctl, fd, TCXONC, action);
  if (ret < 0) {
    libc_errno = -ret;
    return -1;
  }
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
