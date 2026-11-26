//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of pipe.
///
//===----------------------------------------------------------------------===//

#include "src/unistd/pipe.h"

#include "src/__support/OSUtil/linux/syscall_wrappers/pipe.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/sanitizer.h" // for LIBC_MSAN_UNPOISON

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, pipe, (int pipefd[2])) {
  auto result = linux_syscalls::pipe(pipefd);
  LIBC_MSAN_UNPOISON(pipefd, sizeof(int) * 2);
  if (!result) {
    libc_errno = result.error();
    return -1;
  }
  return result.value();
}

} // namespace LIBC_NAMESPACE_DECL
