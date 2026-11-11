//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of mknodat.
///
//===----------------------------------------------------------------------===//

#include "src/sys/stat/mknodat.h"

#include "hdr/types/dev_t.h"
#include "hdr/types/mode_t.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/mknodat.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, mknodat,
                   (int dirfd, const char *path, mode_t mode, dev_t dev)) {
  auto result = linux_syscalls::mknodat(dirfd, path, mode, dev);
  if (!result) {
    libc_errno = result.error();
    return -1;
  }
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
