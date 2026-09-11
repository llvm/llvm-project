//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of the fallocate, which is the
/// fcntl fallocate function for linux.
///
//===----------------------------------------------------------------------===//

#include "src/fcntl/fallocate.h"

#include "src/__support/OSUtil/linux/syscall_wrappers/fallocate.h"
#include "src/__support/macros/config.h"
#include "src/__support/libc_errno.h"
#include "hdr/types/off_t.h"

namespace LIBC_NAMESPACE_DECL {

  LLVM_LIBC_FUNCTION(int, fallocate, (int fd, int mode, off_t offset, off_t size)) {
    auto result = linux_syscalls::fallocate(fd, mode, offset, size);
    if (!result.has_value()) {
      libc_errno = result.error();
      return -1;
    }
    return 0;
  }
}
