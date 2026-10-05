//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of posix_fallocate.
///
//===----------------------------------------------------------------------===//

#include "src/fcntl/posix_fallocate.h"

#include "hdr/errno_macros.h"
#include "hdr/types/off_t.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/fallocate.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, posix_fallocate, (int fd, off_t offset, off_t len)) {
  if (offset < 0 || len <= 0)
    return EINVAL;

  auto result = linux_syscalls::fallocate(fd, 0, offset, len);
  if (!result)
    return result.error();
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
