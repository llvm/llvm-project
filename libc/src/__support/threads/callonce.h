//===-- Types related to the callonce function ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_THREADS_CALLONCE_H
#define LLVM_LIBC_SRC___SUPPORT_THREADS_CALLONCE_H

#include "src/__support/macros/optimization.h"

#ifdef __linux__
#include "src/__support/threads/linux/futex_utils.h"
#else
#error "callonce is not supported on this platform"
#endif

namespace LIBC_NAMESPACE {

#ifdef __linux__
using CallOnceFlag = Futex;
#endif
using CallOnceCallback = void(void);
namespace callonce_impl {
static constexpr FutexWordType NOT_CALLED = 0x0;
static constexpr FutexWordType START = 0x11;
static constexpr FutexWordType WAITING = 0x22;
static constexpr FutexWordType FINISH = 0x33;
int callonce_slowpath(CallOnceFlag *flag, CallOnceCallback *callback);
} // namespace callonce_impl

LIBC_INLINE int callonce(CallOnceFlag *flag, CallOnceCallback *callback) {
  using namespace callonce_impl;
  // Avoid cmpxchg operation if the function has already been called.
  // The destination operand of cmpxchg may receive a write cycle without
  // regard to the result of the comparison
  if (LIBC_LIKELY(flag->load(cpp::MemoryOrder::RELAXED) == FINISH))
    return 0;

  return callonce_slowpath(flag, callback);
}
} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC___SUPPORT_THREADS_CALLONCE_H
