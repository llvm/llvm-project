//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23
// ADDITIONAL_COMPILE_FLAGS: -O0 -g
// XFAIL: availability-stacktrace-missing

/*
  (19.6.4.2)

  // [stacktrace.basic.cons], creation and assignment
  static basic_stacktrace current(const allocator_type& alloc = allocator_type()) noexcept;
*/

#include <cassert>
#include <cstdint>
#include <stacktrace>

uint32_t test1_line;
uint32_t test2_line;
uint32_t main_line;

_LIBCPP_NO_TAIL_CALLS _LIBCPP_NOINLINE std::stacktrace test1() {
  test1_line = __LINE__ + 1; // add 1 to get the next line (where the call to `current` occurs)
  auto ret   = std::stacktrace::current();
  return ret;
}

_LIBCPP_NO_TAIL_CALLS _LIBCPP_NOINLINE std::stacktrace test2() {
  test2_line = __LINE__ + 1; // add 1 to get the next line (where the call to `current` occurs)
  auto ret   = test1();
  return ret;
}

_LIBCPP_NO_TAIL_CALLS _LIBCPP_NOINLINE void test_current() {
  main_line = __LINE__ + 1; // add 1 to get the next line (where the call to `current` occurs)
  auto st   = test2();

  assert(st.size() >= 3);
  assert(st[0]);
  assert(st[0].native_handle());
  // assert(st[0].description().contains("test1"));
  // assert(st[0].source_file().ends_with("current_no_args.pass.cpp"));
  // assert(st[0].source_line() == test1_line);
  assert(st[1]);
  assert(st[1].native_handle());
  // assert(st[1].description().contains("test2"));
  // assert(st[1].source_file().ends_with("current_no_args.pass.cpp"));
  // assert(st[1].source_line() == test2_line);
  assert(st[2]);
  assert(st[2].native_handle());
  // assert(st[2].description().contains("test_current"));
  // assert(st[2].source_file().ends_with("current_no_args.pass.cpp"));
  // assert(st[2].source_line() == main_line);
}

_LIBCPP_NO_TAIL_CALLS
int main(int, char**) {
  static_assert(noexcept(std::stacktrace::current()));
  test_current();
  return 0;
}
