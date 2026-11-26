//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of the memfd_create function.
///
//===----------------------------------------------------------------------===//

#include "src/sys/mman/memfd_create.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/memfd_create.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, memfd_create, (const char *name, unsigned int flags)) {
  ErrorOr<int> result = linux_syscalls::memfd_create(name, flags);
  if (!result) {
    libc_errno = result.error();
    return -1;
  }
  return result.value();
}

} // namespace LIBC_NAMESPACE_DECL
