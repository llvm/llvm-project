//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

//       iterator end(); // constexpr since C++20
// const_iterator end() const; // constexpr since C++20

#include <string>
#include <cassert>
#include <cstddef>

#include "test_macros.h"
#include "min_allocator.h"

template <class S>
TEST_CONSTEXPR_CXX20 void test(S s) {
  const S& cs                   = s;
  typename S::iterator e        = s.end();
  typename S::const_iterator ce = cs.end();
  if (s.empty()) {
    assert(e == s.begin());
    assert(ce == cs.begin());
  }
  assert(static_cast<std::size_t>(e - s.begin()) == s.size());
  assert(static_cast<std::size_t>(ce - cs.begin()) == cs.size());
}

template <class S>
TEST_CONSTEXPR_CXX20 void test_string() {
  test(S());
  test(S("123"));
}

TEST_CONSTEXPR_CXX20 bool test() {
  test_string<std::string>();
#if TEST_STD_VER >= 11
  test_string<std::basic_string<char, std::char_traits<char>, min_allocator<char> > >();
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
