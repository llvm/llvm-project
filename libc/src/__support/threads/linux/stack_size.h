//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Default thread stack size determination for Linux.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_THREADS_LINUX_STACK_SIZE_H
#define LLVM_LIBC_SRC___SUPPORT_THREADS_LINUX_STACK_SIZE_H

#include "hdr/limits_macros.h" // POSIX defines PTHREAD_STACK_MIN in limits.h.
#include "hdr/pthread_macros.h"
#include "hdr/sys_resource_macros.h"
#include "hdr/types/size_t.h"
#include "hdr/types/struct_rlimit.h"
#include "src/__support/CPP/algorithm.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/prlimit.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"

#include <linux/param.h> // For EXEC_PAGESIZE.

namespace LIBC_NAMESPACE_DECL {
namespace detail {

LIBC_INLINE_VAR constexpr size_t FALLBACK_STACKSIZE = 2 * 1024 * 1024;

// Derives the default thread stack size from the soft RLIMIT_STACK limit.
// If the limit is RLIM_INFINITY or too large to page-align in size_t (leaving
// room for the default guard page), falls back to FALLBACK_STACKSIZE.
// Otherwise, clamps to at least PTHREAD_STACK_MIN and rounds up to
// EXEC_PAGESIZE.
LIBC_INLINE size_t stacksize_from_rlimit(rlim_t rlim_cur) {
  constexpr size_t PAGE_MASK = static_cast<size_t>(EXEC_PAGESIZE) - 1;
  constexpr size_t MAX_STACKSIZE =
      ~PAGE_MASK - static_cast<size_t>(EXEC_PAGESIZE);

  if (rlim_cur == RLIM_INFINITY || rlim_cur > MAX_STACKSIZE)
    return FALLBACK_STACKSIZE;

  size_t stacksize = cpp::max(static_cast<size_t>(rlim_cur),
                              static_cast<size_t>(PTHREAD_STACK_MIN));
  return (stacksize + PAGE_MASK) & ~PAGE_MASK;
}

// Written once by init_default_stacksize() during process startup, before any
// other threads exist, and only read afterwards.
LIBC_INLINE_VAR LIBC_CONSTINIT size_t default_stacksize_value =
    FALLBACK_STACKSIZE;

} // namespace detail

LIBC_INLINE void init_default_stacksize() {
  struct rlimit limits;
  if (linux_syscalls::prlimit(0, RLIMIT_STACK, nullptr, &limits).has_value())
    detail::default_stacksize_value =
        detail::stacksize_from_rlimit(limits.rlim_cur);
}

LIBC_INLINE size_t default_stacksize() {
  return detail::default_stacksize_value;
}

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_THREADS_LINUX_STACK_SIZE_H
