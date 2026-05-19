//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of recvfrom.
///
//===----------------------------------------------------------------------===//
#include "src/sys/socket/recvfrom.h"
#include "hdr/types/socklen_t.h"
#include "hdr/types/ssize_t.h"
#include "hdr/types/struct_sockaddr.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/recvfrom.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/sanitizer.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(ssize_t, recvfrom,
                   (int sockfd, void *__restrict buf, size_t len, int flags,
                    sockaddr *__restrict src_addr,
                    socklen_t *__restrict addrlen)) {
  // addrlen is a value-result argument. If it's not null, it passes the max
  // size of the buffer src_addr to the syscall. After the syscall, it's updated
  // to the actual size of the source address. This may be larger than the
  // buffer, in which case the buffer contains a truncated result.
  size_t srcaddr_sz;
  if (src_addr)
    srcaddr_sz = *addrlen;
  (void)srcaddr_sz; // prevent "set but not used" warning

  auto result =
      linux_syscalls::recvfrom(sockfd, buf, len, flags, src_addr, addrlen);
  if (!result.has_value()) {
    libc_errno = result.error();
    return -1;
  }

  ssize_t ret = result.value();
  MSAN_UNPOISON(buf, ret);

  if (src_addr) {
    size_t min_src_addr_size = (*addrlen < srcaddr_sz) ? *addrlen : srcaddr_sz;
    (void)min_src_addr_size; // prevent "set but not used" warning

    MSAN_UNPOISON(src_addr, min_src_addr_size);
  }
  return ret;
}

} // namespace LIBC_NAMESPACE_DECL
