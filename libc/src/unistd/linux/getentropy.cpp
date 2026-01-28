//===-- Linux implementation of getentropy --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/unistd/getentropy.h"
#include "hdr/errno_macros.h"
#include "src/__support/OSUtil/syscall.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"

#include <sys/syscall.h> // For syscall numbers.

namespace LIBC_NAMESPACE_DECL {
LLVM_LIBC_FUNCTION(int, getentropy, (void *buffer, size_t length)) {
  // check the length limit
  if (length > 256) {
    libc_errno = EIO;
    return -1;
  }

  char *cursor = static_cast<char *>(buffer);
  while (length != 0) {
    // 0 flag means urandom and blocking, which meets the assumption of
    // getentropy
    auto ret = syscall_impl<long>(SYS_getrandom, cursor, length, 0);

    // on success, advance the buffer pointer
    if (ret >= 0) {
      length -= static_cast<size_t>(ret);
      cursor += ret;
      continue;
    }

    auto error = -static_cast<int>(ret);

    // on EINTR, try again
    if (error == EINTR)
      continue;

    // on ENOSYS, forward errno and exit;
    // otherwise, set EIO and exit
    libc_errno = (error == ENOSYS) ? ENOSYS : EIO;
    return -1;
  }
  return 0;
}
} // namespace LIBC_NAMESPACE_DECL
