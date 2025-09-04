//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23
// XFAIL: availability-stacktrace-missing

/*
  (19.6.4.2)

  // [stacktrace.basic.cons], creation and assignment
  explicit basic_stacktrace(const allocator_type& alloc) noexcept;
*/

#include <cassert>
#include <stacktrace>
#include <type_traits>

#include "../test_allocs.h"

void test_construct_with_alloc() {
  std::stacktrace st;
  assert(st.empty());
}

void test_construct_with_alloc_noexcept() {
  static_assert(noexcept(std::stacktrace()));

  using A1 = std::allocator<std::stacktrace_entry>;
  static_assert(std::is_nothrow_constructible_v<A1>);
  using A2 = TestAlloc<std::stacktrace_entry, false, true, true, true>;
  static_assert(!std::is_nothrow_constructible_v<A2>);

  static_assert(!noexcept(std::basic_stacktrace<A2>(A2())));
}

int main(int, char**) {
  test_construct_with_alloc();
  test_construct_with_alloc_noexcept();
  return 0;
}
