//===-- Linux implementation of getentropy --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/unistd/getentropy.h"
#include "hdr/errno_macros.h"
#include "src/__support/common.h"
#include "src/errno/libc_errno.h"
#include "src/sys/random/getrandom.h"

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
    ssize_t ret = LIBC_NAMESPACE::getrandom(cursor, length, 0);

    // on success, advance the buffer pointer
    if (ret != -1) {
      length -= static_cast<size_t>(ret);
      cursor += ret;
      continue;
    }

    // on EINTR, try again
    if (libc_errno == EINTR)
      continue;

    // on ENOSYS, forward errno and exit;
    // otherwise, set EIO and exit
    if (libc_errno != ENOSYS)
      libc_errno = EIO;
    return -1;
  }
  return 0;
}
} // namespace LIBC_NAMESPACE_DECL
