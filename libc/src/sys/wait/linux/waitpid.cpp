//===-- Linux implementation of waitpid -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/common.h"
#include "src/__support/libc_assert.h"

#include "src/sys/wait/wait4Impl.h"
#include "src/sys/wait/waitpid.h"

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(pid_t, waitpid, (pid_t pid, int *wait_status, int options)) {
  auto result = internal::wait4impl(pid, wait_status, options, 0);
  if (!result.has_value()) {
    libc_errno = result.error();
    return -1;
  }
  return result.value();
}

} // namespace LIBC_NAMESPACE
