//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// TODO: Change to XFAIL once https://llvm.org/PR40995 is fixed
// UNSUPPORTED: availability-pmr-missing

// check that functions are marked [[nodiscard]]

// [[nodiscard]] std::pmr::memory_resource::allocate(size_t, size_t);
// [[nodiscard]] std::pmr::polymorphic_allocator<T>::allocate(size_t, size_t);

#include <memory_resource>

void f() {
  std::pmr::memory_resource* res = nullptr;
  res->allocate(0);    // expected-warning {{ignoring return value of function}}
  res->allocate(0, 1); // expected-warning {{ignoring return value of function}}

  std::pmr::polymorphic_allocator<int> poly;
  poly.allocate(0); // expected-warning {{ignoring return value of function}}
}
