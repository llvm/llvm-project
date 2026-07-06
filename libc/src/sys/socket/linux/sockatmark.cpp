//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of sockatmark.
///
//===----------------------------------------------------------------------===//

#include "src/sys/socket/sockatmark.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/ioctl.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"

// For SIOCATMARK. Can't use hdr/sys_ioctl_macros, as not all system library
// versions define it.
#include <linux/sockios.h> // For SIOCATMARK.

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, sockatmark, (int sockfd)) {
  int mark = 0;
  ErrorOr<int> result = linux_syscalls::ioctl(sockfd, SIOCATMARK, &mark);
  if (!result.has_value()) {
    libc_errno = result.error();
    return -1;
  }

  return mark;
}

} // namespace LIBC_NAMESPACE_DECL
