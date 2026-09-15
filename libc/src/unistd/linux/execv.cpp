//===-- Linux implementation of execv -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/unistd/execv.h"
#include "src/__support/macros/config.h"
#include "src/unistd/environ.h"

#include "src/__support/OSUtil/linux/syscall_wrappers/execve.h"
#include "src/__support/common.h"

#include "src/__support/libc_errno.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, execv, (const char *path, char *const argv[])) {
  auto ret = linux_syscalls::execve(
      path, argv, const_cast<char *const *>(LIBC_NAMESPACE::environ));
  if (!ret) {
    libc_errno = ret.error();
    return -1;
  }

  // Control will not reach here on success but have a return statement will
  // keep the compilers happy.
  return *ret;
}

} // namespace LIBC_NAMESPACE_DECL
