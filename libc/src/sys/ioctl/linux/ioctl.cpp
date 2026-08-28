//===---------- Linux implementation of the ioctl function ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/ioctl/ioctl.h"

#include "src/__support/OSUtil/linux/syscall_wrappers/ioctl.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include <stdarg.h>

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, ioctl, (int fd, unsigned long request, ...)) {
  va_list vargs;
  va_start(vargs, request);
  void *data_pointer = va_arg(vargs, void *);
  va_end(vargs);

  auto ret = linux_syscalls::ioctl(fd, request, data_pointer);

  if (ret.has_value())
    return ret.value();

  libc_errno = ret.error();
  return -1;
}

} // namespace LIBC_NAMESPACE_DECL
