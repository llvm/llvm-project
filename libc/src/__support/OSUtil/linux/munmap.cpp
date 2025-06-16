//===---- Linux implementation of the POSIX munmap function -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/OSUtil/munmap.h"

#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/common.h"

#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include <sys/syscall.h> // For syscall numbers.

namespace LIBC_NAMESPACE_DECL {
namespace internal {

// This function is currently linux only. It has to be refactored suitably if
// munmap is to be supported on non-linux operating systems also.
int munmap(void *addr, size_t size) {
  return LIBC_NAMESPACE::syscall_impl<int>(SYS_munmap,
                                           reinterpret_cast<long>(addr), size);
}

} // namespace internal
} // namespace LIBC_NAMESPACE_DECL
