//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// void shrink_to_fit(); // constexpr since C++20

#include <string>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

template <class S>
TEST_CONSTEXPR_CXX20 void test(S s) {
  typename S::size_type old_cap = s.capacity();
  S s0                          = s;
  s.shrink_to_fit();
  LIBCPP_ASSERT(s.__invariants());
  assert(s == s0);
  assert(s.capacity() <= old_cap);
  assert(s.capacity() >= s.size());
}

template <class S>
TEST_CONSTEXPR_CXX20 void test_string() {
  S s;
  test(s);

  s.assign(10, 'a');
  s.erase(5);
  test(s);

  s.assign(100, 'a');
  s.erase(50);
  test(s);
}

TEST_CONSTEXPR_CXX20 bool test() {
  test_string<std::string>();
#if TEST_STD_VER >= 11
  test_string<std::basic_string<char, std::char_traits<char>, min_allocator<char>>>();
#endif

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER > 17
  static_assert(test());
#endif

  return 0;
}
