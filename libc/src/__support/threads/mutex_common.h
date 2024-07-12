//===--- Common definitions useful for mutex implementations ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_THREADS_MUTEX_COMMON_H
#define LLVM_LIBC_SRC___SUPPORT_THREADS_MUTEX_COMMON_H

namespace LIBC_NAMESPACE {

enum class MutexError : int {
  NONE,
  BUSY,
  TIMEOUT,
  UNLOCK_WITHOUT_LOCK,
  BAD_LOCK_STATE,
};

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC___SUPPORT_THREADS_MUTEX_COMMON_H
