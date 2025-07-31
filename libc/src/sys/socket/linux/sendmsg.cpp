//===-- Linux implementation of sendmsg -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/socket/sendmsg.h"

#include <linux/net.h>   // For SYS_SOCKET socketcall number.
#include <sys/syscall.h> // For syscall numbers.

#include "hdr/types/ssize_t.h"
#include "hdr/types/struct_msghdr.h"
#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(ssize_t, sendmsg,
                   (int sockfd, const struct msghdr *msg, int flags)) {
#ifdef SYS_sendmsg
  ssize_t ret =
      LIBC_NAMESPACE::syscall_impl<ssize_t>(SYS_sendmsg, sockfd, msg, flags);
#elif defined(SYS_socketcall)
  unsigned long sockcall_args[3] = {static_cast<unsigned long>(sockfd),
                                    reinterpret_cast<unsigned long>(msg),
                                    static_cast<unsigned long>(flags)};
  ssize_t ret = LIBC_NAMESPACE::syscall_impl<ssize_t>(
      SYS_socketcall, SYS_SENDMSG, sockcall_args);
#else
#error "socket and socketcall syscalls unavailable for this platform."
#endif
  if (ret < 0) {
    libc_errno = static_cast<int>(-ret);
    return -1;
  }
  return ret;
}

} // namespace LIBC_NAMESPACE_DECL
