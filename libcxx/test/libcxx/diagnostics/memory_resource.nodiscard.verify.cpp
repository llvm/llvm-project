//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// check that <memory_resource> functions are marked [[nodiscard]]

// clang-format off

#include <memory_resource>

#include "test_macros.h"

void test() {
  std::pmr::memory_resource* resource = std::pmr::null_memory_resource();
  resource->allocate(1); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::pmr::polymorphic_allocator<int> polymorphic_allocator;
  polymorphic_allocator.allocate(1); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}
