//===-- Linux implementation of socketpair --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/socket/socketpair.h"

#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/sanitizer.h"
#include "src/errno/libc_errno.h"
#include <linux/net.h>   // For SYS_SOCKET socketcall number.
#include <sys/syscall.h> // For syscall numbers.

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, socketpair,
                   (int domain, int type, int protocol, int sv[2])) {
#ifdef SYS_socketpair
  int ret = LIBC_NAMESPACE::syscall_impl<int>(SYS_socketpair, domain, type,
                                              protocol, sv);
#elif defined(SYS_socketcall)
  unsigned long sockcall_args[3] = {
      static_cast<unsigned long>(domain), static_cast<unsigned long>(type),
      static_cast<unsigned long>(protocol), static_cast<unsigned long>(sv)};
  int ret = LIBC_NAMESPACE::syscall_impl<int>(SYS_socketcall, SYS_SOCKETPAIR,
                                              sockcall_args);
#else
#error "socket and socketcall syscalls unavailable for this platform."
#endif
  if (ret < 0) {
    libc_errno = -ret;
    return -1;
  }

  MSAN_UNPOISON(sv, sizeof(int) * 2);

  return ret;
}

} // namespace LIBC_NAMESPACE_DECL
