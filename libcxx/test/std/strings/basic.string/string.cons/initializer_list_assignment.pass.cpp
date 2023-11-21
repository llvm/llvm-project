//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <string>

// basic_string& operator=(initializer_list<charT> il); // constexpr since C++20

#include <string>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

// clang-format off
template <template <class> class Alloc>
TEST_CONSTEXPR_CXX20 void test_string() {
  {
    typedef std::basic_string<char, std::char_traits<char>, Alloc<char>> S;
    S s;
    S& result = (s = {'a', 'b', 'c'});
    assert(s == "abc");
    assert(&result == &s);
  }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  {
    typedef std::basic_string<wchar_t, std::char_traits<wchar_t>, Alloc<wchar_t>> S;
    S s;
    S& result = (s = {L'a', L'b', L'c'});
    assert(s == L"abc");
    assert(&result == &s);
  }
#endif
}
// clang-format on

TEST_CONSTEXPR_CXX20 bool test() {
  test_string<std::allocator>();
  test_string<min_allocator>();

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER > 17
  static_assert(test());
#endif

  return 0;
}
