//===--- A simple std::mutex implementation ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_CPP_MUTEX_H
#define LLVM_LIBC_SRC___SUPPORT_CPP_MUTEX_H

// Platform independent code will include this header file which pulls
// the platfrom specific specializations using platform macros.
//
// The platform specific specializations should define a class by name
// Mutex with non-static methods having the following signature:
//
// MutexError lock();
// MutexError trylock();
// MutexError timedlock(...);
// MutexError unlock();
// MutexError reset(); // Used to reset inconsistent robust mutexes.
//
// Apart from the above non-static methods, the specializations should
// also provide few static methods with the following signature:
//
// static MutexError init(mtx_t *);
// static MutexError destroy(mtx_t *);
//
// All of the static and non-static methods should ideally be implemented
// as inline functions so that implementations of public functions can
// call them without a function call overhead.
//
// Another point to keep in mind that is that the libc internally needs a
// few global locks. So, to avoid static initialization order fiasco, we
// want the constructors of the Mutex classes to be constexprs.

namespace LIBC_NAMESPACE {
namespace cpp {

// An RAII class for easy locking and unlocking of mutexes.
template<typename mutex_type>
class lock_guard {
public:
  explicit lock_guard(mutex_type &m) : mutex(m) { mutex->lock(); }

  ~lock_guard() { mutex->unlock(); }

  // non-copyable
  lock_guard &operator=(const lock_guard &) = delete;
  lock_guard(const lock_guard &) = delete;

private:
  mutex_type &mutex;
};

} // namespace cpp
} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC___SUPPORT_CPP_MUTEX_H
