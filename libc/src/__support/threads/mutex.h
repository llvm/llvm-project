//===--- A platform independent abstraction layer for mutexes ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_THREADS_MUTEX_H
#define LLVM_LIBC_SRC___SUPPORT_THREADS_MUTEX_H

#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"

#if LIBC_THREAD_MODE == LIBC_THREAD_MODE_PLATFORM

// Platform independent code will include this header file which pulls
// the platform specific specializations using platform macros.
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

#if defined(__linux__)
#include "src/__support/threads/linux/mutex.h"
#endif // __linux__

#elif LIBC_THREAD_MODE == LIBC_THREAD_MODE_SINGLE

#include "src/__support/threads/mutex_common.h"

namespace LIBC_NAMESPACE_DECL {

/// Implementation of a simple passthrough mutex which guards nothing. A
/// complete Mutex locks in general cannot be implemented on the GPU, or on some
/// baremetal platforms. We simply define the Mutex interface and require that
/// only a single thread executes code requiring a mutex lock.
struct Mutex {
  LIBC_INLINE constexpr Mutex(bool, bool, bool, bool) {}

  LIBC_INLINE MutexError lock() { return MutexError::NONE; }
  LIBC_INLINE MutexError unlock() { return MutexError::NONE; }
  LIBC_INLINE MutexError reset() { return MutexError::NONE; }
};

} // namespace LIBC_NAMESPACE_DECL

#elif LIBC_THREAD_MODE == LIBC_THREAD_MODE_EXTERNAL

// TODO: Implement the interfacing, if necessary, e.g. "extern struct Mutex;"

#endif // LIBC_THREAD_MODE == LIBC_THREAD_MODE_PLATFORM

#endif // LLVM_LIBC_SRC___SUPPORT_THREADS_MUTEX_H
