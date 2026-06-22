//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of sendmmsg.
///
//===----------------------------------------------------------------------===//
#include "src/sys/socket/sendmmsg.h"
#include "hdr/types/struct_mmsghdr.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/sendmmsg.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, sendmmsg,
                   (int sockfd, struct mmsghdr *msgvec, unsigned int vlen,
                    int flags)) {
  auto result = linux_syscalls::sendmmsg(sockfd, msgvec, vlen, flags);
  if (!result.has_value()) {
    libc_errno = result.error();
    return -1;
  }
  return result.value();
}

} // namespace LIBC_NAMESPACE_DECL
