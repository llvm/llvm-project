//===---------- Linux implementation of the shm_unlink function -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/mman/shm_unlink.h"

#include "hdr/fcntl_macros.h"
#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/libc_errno.h"     // For internal errno.
#include "src/__support/macros/config.h"
#include "src/sys/mman/linux/shm_common.h"
#include <sys/syscall.h> // For SYS_unlink, SYS_unlinkat

namespace LIBC_NAMESPACE_DECL {

// TODO: move the unlink syscall to a shared utility.

LLVM_LIBC_FUNCTION(int, shm_unlink, (const char *name)) {
  auto path_result = shm_common::translate_name(name);
  if (!path_result.has_value()) {
    libc_errno = path_result.error();
    return -1;
  }
#ifdef SYS_unlink
  int ret = LIBC_NAMESPACE::syscall_impl<int>(SYS_unlink, path_result->data());
#elif defined(SYS_unlinkat)
  int ret = LIBC_NAMESPACE::syscall_impl<int>(SYS_unlinkat, AT_FDCWD,
                                              path_result->data(), 0);
#else
#error "unlink and unlinkat syscalls not available."
#endif

  if (ret < 0) {
    libc_errno = -ret;
    return -1;
  }
  return ret;
}

} // namespace LIBC_NAMESPACE_DECL
