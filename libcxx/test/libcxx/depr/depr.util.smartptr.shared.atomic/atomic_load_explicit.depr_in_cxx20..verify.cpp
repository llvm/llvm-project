//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_ENABLE_CXX26_REMOVED_SHARED_PTR_ATOMICS
// UNSUPPORTED: no-threads
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <memory>

// shared_ptr

// template <class T>
// shared_ptr<T>
// atomic_load_explicit(const shared_ptr<T>* p, memory_order mo)     // Deprecated in C++20, removed in C++26

#include <atomic>
#include <memory>
#include <tuple>

void test() {
  const std::shared_ptr<int> p(new int(3));
  // expected-warning@+1 {{'atomic_load_explicit<int>' is deprecated}}
  std::ignore = std::atomic_load_explicit(&p, std::memory_order_relaxed);
}
