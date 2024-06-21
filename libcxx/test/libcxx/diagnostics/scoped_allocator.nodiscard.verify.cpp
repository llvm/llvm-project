//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// check that <scoped_allocator> functions are marked [[nodiscard]]

// clang-format off

#include <memory>
#include <scoped_allocator>

void test() {
  std::scoped_allocator_adaptor<std::allocator<int>> alloc;
  alloc.allocate(1);          // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  alloc.allocate(1, nullptr); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}
