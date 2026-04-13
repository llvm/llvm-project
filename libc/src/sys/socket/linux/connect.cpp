//===-- Linux implementation of connect -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/socket/connect.h"

#include "hdr/types/socklen_t.h"
#include "hdr/types/struct_sockaddr.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/connect.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, connect,
                   (int sockfd, const struct sockaddr *addr,
                    socklen_t addrlen)) {
  auto result = linux_syscalls::connect(sockfd, addr, addrlen);
  if (!result.has_value()) {
    libc_errno = result.error();
    return -1;
  }

  return result.value();
}

} // namespace LIBC_NAMESPACE_DECL
