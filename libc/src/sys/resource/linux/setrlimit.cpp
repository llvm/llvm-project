//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of setrlimit.
///
//===----------------------------------------------------------------------===//

#include "src/sys/resource/setrlimit.h"

#include "hdr/types/struct_rlimit.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/prlimit.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, setrlimit, (int res, const struct rlimit *limits)) {
  auto result = linux_syscalls::prlimit(0, res, limits, nullptr);
  if (!result) {
    libc_errno = result.error();
    return -1;
  }
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
