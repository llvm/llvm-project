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
// void
// atomic_store(shared_ptr<T>* p, shared_ptr<T> r)   // Deprecated in C++20, removed in C++26

#include <memory>

void test() {
  std::shared_ptr<int> p;
  std::shared_ptr<int> r(new int(3));
  // expected-warning@+1 {{'atomic_store<int>' is deprecated}}
  std::atomic_store(&p, r);
}
