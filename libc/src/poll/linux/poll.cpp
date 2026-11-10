//===-- Implementation of poll --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/poll/poll.h"

#include "hdr/types/nfds_t.h"
#include "hdr/types/struct_pollfd.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/poll.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, poll, (pollfd * fds, nfds_t nfds, int timeout)) {
  auto result = linux_syscalls::poll(fds, nfds, timeout);
  if (!result) {
    libc_errno = result.error();
    return -1;
  }
  return result.value();
}

} // namespace LIBC_NAMESPACE_DECL
