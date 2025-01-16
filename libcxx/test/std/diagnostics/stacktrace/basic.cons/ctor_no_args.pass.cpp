//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23
// ADDITIONAL_COMPILE_FLAGS: -g

/*
  (19.6.4.2)

  // [stacktrace.basic.cons], creation and assignment
  basic_stacktrace() noexcept(is_nothrow_default_constructible_v<allocator_type>);
*/

#include <cassert>
#include <stacktrace>

uint32_t test1_line;
uint32_t test2_line;

template <class A>
_LIBCPP_NO_TAIL_CALLS _LIBCPP_NOINLINE std::basic_stacktrace<A> test1(A& alloc) {
  test1_line = __LINE__ + 1; // add 1 to get the next line (where the call to `current` occurs)
  auto ret   = std::basic_stacktrace<A>::current(alloc);
  return ret;
}

template <class A>
_LIBCPP_NO_TAIL_CALLS _LIBCPP_NOINLINE std::basic_stacktrace<A> test2(A& alloc) {
  test2_line = __LINE__ + 1; // add 1 to get the next line (where the call to `current` occurs)
  auto ret   = test1(alloc);
  return ret;
}

_LIBCPP_NO_TAIL_CALLS _LIBCPP_NOINLINE void test_default_construct() {
  std::stacktrace st;
  assert(st.empty());
}

_LIBCPP_NO_TAIL_CALLS
int main(int, char**) {
  test_default_construct();
  return 0;
}
