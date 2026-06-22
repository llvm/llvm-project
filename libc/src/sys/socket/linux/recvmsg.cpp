//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of recvmsg.
///
//===----------------------------------------------------------------------===//
#include "src/sys/socket/recvmsg.h"
#include "hdr/types/ssize_t.h"
#include "hdr/types/struct_msghdr.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/recvmsg.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/sanitizer.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(ssize_t, recvmsg, (int sockfd, msghdr *msg, int flags)) {
  auto result = linux_syscalls::recvmsg(sockfd, msg, flags);
  if (!result.has_value()) {
    libc_errno = result.error();
    return -1;
  }

  // Unpoison the msghdr, as well as all its components.
  MSAN_UNPOISON(msg, sizeof(msghdr));
  MSAN_UNPOISON(msg->msg_name, msg->msg_namelen);

  for (size_t i = 0; i < msg->msg_iovlen; ++i) {
    MSAN_UNPOISON(msg->msg_iov[i].iov_base, msg->msg_iov[i].iov_len);
  }
  MSAN_UNPOISON(msg->msg_control, msg->msg_controllen);

  return result.value();
}

} // namespace LIBC_NAMESPACE_DECL
