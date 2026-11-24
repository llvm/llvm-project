//===---------- Linux implementation of the shm_unlink function -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/mman/shm_unlink.h"

#include "src/__support/OSUtil/linux/syscall_wrappers/unlink.h"
#include "src/__support/libc_errno.h" // For internal errno.
#include "src/__support/macros/config.h"
#include "src/sys/mman/linux/shm_common.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, shm_unlink, (const char *name)) {
  auto path_result = shm_common::translate_name(name);
  if (!path_result.has_value()) {
    libc_errno = path_result.error();
    return -1;
  }

  auto result = linux_syscalls::unlink(path_result->data());
  if (!result) {
    libc_errno = result.error();
    return -1;
  }
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
