//===-- Linux implementation of posix_close -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/unistd/posix_close.h"

#include "hdr/unistd_macros.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/close.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, posix_close, (int fd, int flag)) {
#ifdef POSIX_CLOSE_RESTART
  if (flag != 0 && flag != POSIX_CLOSE_RESTART) {
#else
  if (flag != 0) {
#endif
    libc_errno = EINVAL;
    return -1;
  }

  auto result = linux_syscalls::close(fd);

  if (!result.has_value()) {
    libc_errno = result.error();
    return -1;
  }
  return result.value();
}

} // namespace LIBC_NAMESPACE_DECL
