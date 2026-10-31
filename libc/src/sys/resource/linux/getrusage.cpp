//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of getrusage.
///
//===----------------------------------------------------------------------===//

#include "src/sys/resource/getrusage.h"

#include "hdr/types/struct_rusage.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/getrusage.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, getrusage, (int who, struct rusage *usage)) {
  auto result = linux_syscalls::getrusage(who, usage);
  if (!result) {
    libc_errno = result.error();
    return -1;
  }
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
