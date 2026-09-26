//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of the getentropy function.
///
//===----------------------------------------------------------------------===//

#include "src/sys/random/getentropy.h"

#include "hdr/errno_macros.h"
#include "hdr/types/size_t.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/getrandom.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, getentropy, (void *buffer, size_t length)) {
  constexpr size_t MAX_BYTES = 256;
  if (length > MAX_BYTES) {
    libc_errno = EIO;
    return -1;
  }

  auto *buf_ptr = reinterpret_cast<char *>(buffer);
  size_t remaining = length;
  while (remaining > 0) {
    auto result = linux_syscalls::getrandom(buf_ptr, remaining, 0);
    if (!result.has_value()) {
      if (result.error() == EINTR)
        continue;
      libc_errno = static_cast<int>(result.error());
      return -1;
    }
    ssize_t bytes_read = result.value();
    buf_ptr += bytes_read;
    remaining -= static_cast<size_t>(bytes_read);
  }
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
