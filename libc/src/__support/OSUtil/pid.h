//===------------ pid_t utilities -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_OSUTIL_PID_H
#define LLVM_LIBC_SRC___SUPPORT_OSUTIL_PID_H
#include "hdr/types/pid_t.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/optimization.h"

#ifndef LIBC_COPT_ENABLE_PID_CACHE
#define LIBC_COPT_ENABLE_PID_CACHE 1
#endif

namespace LIBC_NAMESPACE_DECL {

class ProcessIdentity {
  static LIBC_INLINE_VAR thread_local bool fork_inflight = true;
  static pid_t cache;
  static pid_t get_uncached();

public:
  LIBC_INLINE static void start_fork() { fork_inflight = true; }
  LIBC_INLINE static void end_fork() { fork_inflight = false; }
  LIBC_INLINE static void refresh_cache() { cache = get_uncached(); }
  LIBC_INLINE static pid_t get() {
#if LIBC_COPT_ENABLE_PID_CACHE
    if (LIBC_LIKELY(!fork_inflight))
      return cache;
#endif
    return get_uncached();
  }
};

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_OSUTIL_PID_H
