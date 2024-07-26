//===--- Tid wrapper --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_THREADS_TID_H
#define LLVM_LIBC_SRC___SUPPORT_THREADS_TID_H

// This header is for internal usage which automatically dispatches full build
// and overlay build behaviors.

#include "hdr/types/pid_t.h"
#include "src/__support/common.h"
#ifdef LIBC_FULL_BUILD
#include "src/__support/threads/thread.h"
#else
#include "src/__support/OSUtil/syscall.h"
#include <sys/syscall.h>
#endif // LIBC_FULL_BUILD

namespace LIBC_NAMESPACE_DECL {
LIBC_INLINE pid_t gettid_inline() {
#ifdef LIBC_FULL_BUILD
  return self.get_tid();
#else
  return syscall_impl<pid_t>(SYS_gettid);
#endif
}
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_THREADS_TID_H
