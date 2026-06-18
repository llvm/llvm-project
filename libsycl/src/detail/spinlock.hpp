//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of a lightweight spin lock.
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL_SPINLOCK
#define _LIBSYCL_SPINLOCK

#include <sycl/__impl/detail/config.hpp>

#include <atomic>
#include <thread>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace detail {

/// SpinLock is a synchronization primitive based on std::atomic_flag.
/// It is trivially constructible and suitable for global object wrappers.
class SpinLock {
public:
  bool try_lock() { return !MLock.test_and_set(std::memory_order_acquire); }

  void lock() {
    while (MLock.test_and_set(std::memory_order_acquire))
      std::this_thread::yield();
  }

  void unlock() { MLock.clear(std::memory_order_release); }

private:
  std::atomic_flag MLock = ATOMIC_FLAG_INIT;
};

} // namespace detail

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL_SPINLOCK
