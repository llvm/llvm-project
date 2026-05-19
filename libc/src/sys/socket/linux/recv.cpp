//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of recv.
///
//===----------------------------------------------------------------------===//
#include "src/sys/socket/recv.h"
#include "hdr/types/ssize_t.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/recvfrom.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/sanitizer.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(ssize_t, recv,
                   (int sockfd, void *buf, size_t len, int flags)) {
  auto result =
      linux_syscalls::recvfrom(sockfd, buf, len, flags, nullptr, nullptr);
  if (!result.has_value()) {
    libc_errno = result.error();
    return -1;
  }

  MSAN_UNPOISON(buf, result.value());

  return result.value();
}

} // namespace LIBC_NAMESPACE_DECL
