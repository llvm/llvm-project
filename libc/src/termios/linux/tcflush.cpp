//===-- Linux implementation of tcflush -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/termios/tcflush.h"

#include "src/__support/OSUtil/syscall.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/errno/libc_errno.h"

#include <asm/ioctls.h>  // Safe to include without the risk of name pollution.
#include <sys/syscall.h> // For syscall numbers
#include <termios.h>

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, tcflush, (int fd, int queue_selector)) {
  return LIBC_NAMESPACE::ioctl(fd, TCFLSH, queue_selector);
}

} // namespace LIBC_NAMESPACE_DECL
