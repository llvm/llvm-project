//===-- Linux implementation of socket ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/socket/socket.h"

#include "src/__support/OSUtil/linux/syscall.h" // syscall_impl
#include "src/__support/OSUtil/linux/syscall_wrappers/socketcall.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"

#include <linux/net.h>   // For SYS_SOCKET socketcall number.
#include <sys/syscall.h> // For syscall numbers.

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, socket, (int domain, int type, int protocol)) {
#ifdef SYS_socket
  int ret = syscall_impl<int>(SYS_socket, domain, type, protocol);
#elif defined(SYS_socketcall)
  int ret = socketcall(SYS_SOCKET, domain, type, protocol);
#else
#error "socket and socketcall syscalls unavailable for this platform."
#endif
  if (ret < 0) {
    libc_errno = -ret;
    return -1;
  }
  return ret;
}

} // namespace LIBC_NAMESPACE_DECL
