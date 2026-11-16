//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of pselect.
///
//===----------------------------------------------------------------------===//

#include "src/sys/select/pselect.h"

#include "hdr/types/fd_set.h"
#include "hdr/types/sigset_t.h"
#include "hdr/types/struct_timespec.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/pselect6.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, pselect,
                   (int nfds, fd_set *__restrict readfds,
                    fd_set *__restrict writefds, fd_set *__restrict exceptfds,
                    const struct timespec *__restrict timeout,
                    const sigset_t *__restrict sigmask)) {
  // The Linux raw pselect6 syscall modifies its timeout argument. To conform to
  // POSIX (which declares timeout as const), we pass a copy.
  timespec ts;
  timespec *tsp = nullptr;
  if (timeout != nullptr) {
    ts = *timeout;
    tsp = &ts;
  }
  auto result = linux_syscalls::pselect6(nfds, readfds, writefds, exceptfds,
                                         tsp, sigmask);
  if (!result.has_value()) {
    libc_errno = result.error();
    return -1;
  }
  return result.value();
}

} // namespace LIBC_NAMESPACE_DECL
