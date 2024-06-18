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
namespace LIBC_NAMESPACE {

class ProcessIdentity {
  static pid_t cache;
  static pid_t get_uncached();

public:
  LIBC_INLINE static void invalidate_cache() { cache = -1; }
  LIBC_INLINE static void refresh_cache() { cache = get_uncached(); }
  LIBC_INLINE static pid_t get() {
    if (LIBC_UNLIKELY(cache < 0))
      return get_uncached();
    return cache;
  }
};

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC___SUPPORT_OSUTIL_PID_H
