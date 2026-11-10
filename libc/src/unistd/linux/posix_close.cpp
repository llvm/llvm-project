//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of posix_close.
///
//===----------------------------------------------------------------------===//

#include "src/unistd/posix_close.h"

#include "hdr/errno_macros.h"
#include "hdr/unistd_macros.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/close.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, posix_close, (int fd, int flag)) {
  static_assert(POSIX_CLOSE_RESTART == 0);

  auto result = linux_syscalls::close(fd);

  if (!result.has_value()) {
    int error = result.error();
    // If POSIX_CLOSE_RESTART is zero, posix_close shall not return an EINTR
    // error. On Linux, the descriptor is released immediately, but background
    // state flush may be interrupted. Map EINTR to EINPROGRESS per POSIX.
    if (error == EINTR)
      error = EINPROGRESS;
    libc_errno = error;
    return -1;
  }

  if (flag != 0) {
    libc_errno = EINVAL;
    return -1;
  }

  return result.value();
}

} // namespace LIBC_NAMESPACE_DECL
