//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of eventfd_write.
///
//===----------------------------------------------------------------------===//

#include "src/sys/eventfd/eventfd_write.h"

#include "hdr/errno_macros.h"
#include "hdr/types/eventfd_t.h"
#include "hdr/types/ssize_t.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/write.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, eventfd_write, (int fd, eventfd_t value)) {
  ErrorOr<ssize_t> ret = linux_syscalls::write(fd, &value, sizeof(eventfd_t));
  if (!ret) {
    libc_errno = ret.error();
    return -1;
  }
  if (ret.value() != sizeof(eventfd_t)) {
    libc_errno = EINVAL;
    return -1;
  }
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
