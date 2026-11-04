//===-- Linux implementation of getsockopt --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/socket/getsockopt.h"

#include "hdr/types/socklen_t.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/getsockopt.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, getsockopt,
                   (int sockfd, int level, int optname, void *__restrict optval,
                    socklen_t *__restrict optlen)) {
  auto result =
      linux_syscalls::getsockopt(sockfd, level, optname, optval, optlen);
  if (!result.has_value()) {
    libc_errno = result.error();
    return -1;
  }

  return result.value();
}

} // namespace LIBC_NAMESPACE_DECL
