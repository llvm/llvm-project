//===-- Linux implementation of setsockopt --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/socket/setsockopt.h"

#include "hdr/types/socklen_t.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/setsockopt.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, setsockopt,
                   (int sockfd, int level, int optname, const void *optval,
                    socklen_t optlen)) {
  auto result =
      linux_syscalls::setsockopt(sockfd, level, optname, optval, optlen);
  if (!result.has_value()) {
    libc_errno = result.error();
    return -1;
  }

  return result.value();
}

} // namespace LIBC_NAMESPACE_DECL
