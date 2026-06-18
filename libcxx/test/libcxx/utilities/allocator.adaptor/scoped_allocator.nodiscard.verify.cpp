//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++11

// <scoped_allocator>

// Check that functions are marked [[nodiscard]]

#include <memory>
#include <scoped_allocator>

void test() {
  std::scoped_allocator_adaptor<std::allocator<int>> alloc;
  const std::scoped_allocator_adaptor<std::allocator<int>> cAlloc;

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  alloc.inner_allocator();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  cAlloc.inner_allocator();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  alloc.outer_allocator();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  cAlloc.outer_allocator();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  alloc.allocate(1);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  alloc.allocate(1, nullptr);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  cAlloc.max_size();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  cAlloc.select_on_container_copy_construction();
}
