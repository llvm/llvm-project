//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of fexecve.
///
//===----------------------------------------------------------------------===//

#include "src/unistd/fexecve.h"

#include "hdr/fcntl_macros.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/execveat.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, fexecve,
                   (int fd, char *const argv[], char *const envp[])) {
  auto ret = linux_syscalls::execveat(fd, "", argv, envp, AT_EMPTY_PATH);
  if (!ret.has_value()) {
    libc_errno = ret.error();
    return -1;
  }

  // Control will not reach here on success but having a return statement will
  // keep the compilers happy.
  return *ret;
}

} // namespace LIBC_NAMESPACE_DECL
