//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of the fallocate, which is the
/// base syscall wrapper for all fallocate dependent calls.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_FALLOCATE_H
#define LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_FALLOCATE_H

#include "hdr/errno_macros.h"
#include "hdr/types/off_t.h"
#include "src/__support/OSUtil/linux/syscall.h" // For syscall_checked
#include "src/__support/common.h"
#include "src/__support/error_or.h"
#include "src/__support/macros/config.h"
#include <sys/syscall.h> // For syscall numbers

namespace LIBC_NAMESPACE_DECL {
namespace linux_syscalls {

LIBC_INLINE ErrorOr<int> fallocate(int fd, int mode, off_t offset, off_t size) {
#ifdef SYS_fallocate
#if !__SIZEOF__POINTER == 8 // 64 bit machines
  /* TODO: Add support for 32 bits */
  return Error(ENOSYS);
#else
  return syscall_checked<int>(SYS_fallocate, fd, mode, offset, size);
#endif
#endif
}

} // namespace linux_syscalls
} // namespace LIBC_NAMESPACE_DECL
#endif
