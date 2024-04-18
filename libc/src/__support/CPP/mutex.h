//===--- A simple lock_guard implementation ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_CPP_MUTEX_H
#define LLVM_LIBC_SRC___SUPPORT_CPP_MUTEX_H

namespace LIBC_NAMESPACE {
namespace cpp {

// An RAII class for easy locking and unlocking of mutexes.
template <typename LockableType>
class lock_guard {
public:
  explicit lock_guard(LockableType &m) : mutex(m) { mutex.lock(); }
  ~lock_guard() { mutex.unlock(); }

  // non-copyable
  lock_guard &operator=(const lock_guard &) = delete;
  lock_guard(const lock_guard &) = delete;

private:
  LockableType &mutex;
};

} // namespace cpp
} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC___SUPPORT_CPP_MUTEX_H
