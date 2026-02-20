//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Check that libc++ honors when __SANITIZER_DISABLE_CONTAINER_OVERFLOW__ is set
// and disables the container overflow checks.
//
// REQUIRES: asan
// ADDITIONAL_COMPILE_FLAGS: -D__SANITIZER_DISABLE_CONTAINER_OVERFLOW__

// When libc++ is built with ASAN instrumentation, we can't turn off the ASAN checks,
// and that is diagnosed as an error.
// XFAIL: libcpp-instrumented-with-asan

// std::basic_string::data is const util C++17
// UNSUPPORTED: c++03, c++11, c++14

// The protocol checked by this test is specific to Clang and compiler-rt
// UNSUPPORTED: gcc

#include <deque>
#include <string>
#include <vector>

// This check is somewhat weak because it would pass if we renamed the libc++-internal
// macro and forgot to update this test. But it doesn't hurt to check it in addition to
// the tests below.
#if _LIBCPP_ENABLE_ASAN_CONTAINER_CHECKS
#  error "Container overflow checks should be disabled in libc++"
#endif

void vector() {
  std::vector<int> v;
  v.reserve(100);
  int* data = v.data();

  // This is illegal with respect to std::vector, but legal from the core language perspective since
  // we do own that allocated memory and `int` is an implicit lifetime type. If container overflow
  // checks are enabled, this would fail.
  data[4] = 42;
}

// For std::string, we must use a custom char_traits class to reliably test this behavior. Since
// std::string is externally instantiated in the built library, __SANITIZER_DISABLE_CONTAINER_OVERFLOW__
// will not be honored for any function that happens to be in the built library. Using a custom
// char_traits class ensures that this doesn't get in the way.
struct my_char_traits : std::char_traits<char> {};

void string() {
  std::basic_string<char, my_char_traits> s;
  s.reserve(100);
  char* data = s.data();
  data[4]    = 'x';
}

void deque() {
  std::deque<int> d;
  d.push_back(1);
  d.push_back(2);
  d.push_back(3);
  int* last_element = &d[2];
  d.pop_back();

  // This reference is technically invalidated according to the library. However since
  // we know std::deque is implemented using segments of a fairly large size and we know
  // the non-erased elements are not invalidated by pop_front() (per the Standard), we can
  // rely on the fact that the last element still exists in memory that is owned by the
  // std::deque.
  //
  // If container overflow checks were enabled, this would obviously fail.
  *last_element = 42;
}

int main(int, char**) {
  vector();
  string();
  deque();
  return 0;
}
