//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of recvmmsg.
///
//===----------------------------------------------------------------------===//

#include "src/sys/socket/recvmmsg.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/recvmmsg.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, recvmmsg,
                   (int sockfd, struct mmsghdr *msgvec, unsigned int vlen,
                    int flags, struct timespec *timeout)) {
  auto result = linux_syscalls::recvmmsg(sockfd, msgvec, vlen, flags, timeout);
  if (!result.has_value()) {
    libc_errno = result.error();
    return -1;
  }
  return result.value();
}

} // namespace LIBC_NAMESPACE_DECL
