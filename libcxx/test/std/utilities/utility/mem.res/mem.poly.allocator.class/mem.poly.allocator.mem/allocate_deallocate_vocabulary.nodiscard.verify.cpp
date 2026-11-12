//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// check that functions are marked [[nodiscard]]

// UNSUPPORTED: c++03, c++11, c++14, c++17

// test_memory_resource requires RTTI for dynamic_cast
// UNSUPPORTED: no-rtti

// <memory_resource>

// polymorphic_allocator::allocate_bytes()
// polymorphic_allocator::allocate_object()
// polymorphic_allocator::new_object()

#include <memory_resource>

void func() {
  std::pmr::polymorphic_allocator<> allocator;
  allocator.allocate_bytes(1); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  allocator.allocate_object<int>(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  allocator.new_object<int>(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}
