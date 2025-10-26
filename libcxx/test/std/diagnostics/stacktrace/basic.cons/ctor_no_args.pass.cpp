//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23
// XFAIL: availability-stacktrace-missing

// (19.6.4.2): [stacktrace.basic.cons], creation and assignment
// basic_stacktrace() noexcept(is_nothrow_default_constructible_v<allocator_type>);

#include <cassert>
#include <stacktrace>
#include <type_traits>

#include "test_macros.h"
#include "../test_allocs.h"

void test_default_construct() {
  std::stacktrace st;
  assert(st.empty()); // Postconditions: empty() is true.

  using A1 = std::allocator<std::stacktrace_entry>;
  assert(std::basic_stacktrace<A1>().empty());
}

void test_default_construct_noexcept() {
  static_assert(noexcept(std::stacktrace()));

  using A1 = std::allocator<std::stacktrace_entry>;
  static_assert(std::is_nothrow_default_constructible_v<A1>); // std::allocator is noexcept constructible
  static_assert(noexcept(std::basic_stacktrace<A1>()));       // therefore this ctor should also be noexcept

  using A2 =
      TestAlloc<std::stacktrace_entry,
                /*_KNoExCtors=*/false,
                // Don't care about the following:
                /*_KNoExAlloc=*/true,
                /*_KPropagate=*/true,
                /*_KAlwaysEqual=*/true>;
  static_assert(!std::is_nothrow_default_constructible_v<A2>); // A2 is not noexcept constructible
  static_assert(!noexcept(std::basic_stacktrace<A2>()));       // this ctor should not be noexcept
}

int main(int, char**) {
  test_default_construct();
  test_default_construct_noexcept();
  return 0;
}
