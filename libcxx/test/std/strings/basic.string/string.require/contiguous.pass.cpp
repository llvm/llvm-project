//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <array>

// A string is a contiguous container

#include <string>
#include <cassert>

#include "test_macros.h"
#include "test_allocator.h"
#include "min_allocator.h"

template <class C>
TEST_CONSTEXPR_CXX20 void test_contiguous(const C& c) {
  for (std::size_t i = 0; i < c.size(); ++i)
    assert(*(c.begin() + static_cast<typename C::difference_type>(i)) == *(std::addressof(*c.begin()) + i));
}

template <class Alloc>
TEST_CONSTEXPR_CXX20 void test_string(const Alloc& a) {
  typedef std::basic_string<char, std::char_traits<char>, Alloc> S;

  {
    test_contiguous(S());
    test_contiguous(S("1"));
    test_contiguous(S("1234567890123456789012345678901234567890123456789012345678901234567890"));
  }
  {
    test_contiguous(S(Alloc()));
    test_contiguous(S("1", Alloc()));
    test_contiguous(S("1234567890123456789012345678901234567890123456789012345678901234567890", Alloc()));
  }
  {
    test_contiguous(S(Alloc(a)));
    test_contiguous(S("1", Alloc(a)));
    test_contiguous(S("1234567890123456789012345678901234567890123456789012345678901234567890", Alloc(a)));
  }
}

TEST_CONSTEXPR_CXX20 bool test() {
  test_string(std::allocator<char>());
  test_string(test_allocator<char>());
  test_string(test_allocator<char>(3));
#if TEST_STD_VER >= 11
  test_string(min_allocator<char>());
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
