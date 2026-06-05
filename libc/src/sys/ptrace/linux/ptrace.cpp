//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of the ptrace function.
///
//===----------------------------------------------------------------------===//

#include "src/sys/ptrace/ptrace.h"

#include "src/__support/OSUtil/linux/syscall_wrappers/ptrace.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(long, ptrace, (int request, ...)) {
  // TODO: Handle request arguments.
  auto result = linux_syscalls::ptrace(request);

  if (!result.has_value()) {
    libc_errno = result.error();
    return -1;
  }

  return result.value();
}

} // namespace LIBC_NAMESPACE_DECL
