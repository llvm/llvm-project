//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

/*
  (19.6.4.2)

  // [stacktrace.basic.cons], creation and assignment
  basic_stacktrace() noexcept(is_nothrow_default_constructible_v<allocator_type>);
*/

#include <cassert>
#include <stacktrace>
#include <type_traits>

#include "../test_allocs.h"

void test_default_construct() {
  std::stacktrace st;
  assert(st.empty());
}

void test_default_construct_noexcept() {
  static_assert(noexcept(std::stacktrace()));

  using A1 = std::allocator<std::stacktrace_entry>;
  static_assert(std::is_nothrow_default_constructible_v<A1>);
  static_assert(noexcept(std::basic_stacktrace<A1>()));

  using A2 = TestAlloc<std::stacktrace_entry, false, true, true, true>;
  static_assert(!std::is_nothrow_default_constructible_v<A2>);
  static_assert(!noexcept(std::basic_stacktrace<A2>()));
}

int main(int, char**) {
  test_default_construct();
  test_default_construct_noexcept();
  return 0;
}
