//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// template<class charT, class traits, class Allocator>
//   basic_string<charT,traits,Allocator>
//   operator+(charT lhs, const basic_string<charT,traits,Allocator>& rhs); // constexpr since C++20

// template<class charT, class traits, class Allocator>
//   basic_string<charT,traits,Allocator>&&
//   operator+(charT lhs, basic_string<charT,traits,Allocator>&& rhs); // constexpr since C++20

#include <cassert>
#include <string>
#include <utility>

#include "test_macros.h"
#include "min_allocator.h"
#include "asan_testing.h"

template <class S>
TEST_CONSTEXPR_CXX20 void test_string() {
  const char* test_data[] = {"", "12345", "1234567890", "12345678901234567890"};
  const char* results[]   = {"a", "a12345", "a1234567890", "a12345678901234567890"};

  for (size_t i = 0; i != 4; ++i) {
    { // operator+(value_type, const string&);
      const S str(test_data[i]);
      assert('a' + str == results[i]);
      LIBCPP_ASSERT(is_string_asan_correct('a' + str));
    }
#if TEST_STD_VER >= 11
    { // operator+(value_type, string&&);
      S str(test_data[i]);
      assert('a' + std::move(str) == results[i]);
    }
#endif
  }
}

TEST_CONSTEXPR_CXX20 bool test() {
  test_string<std::string>();
#if TEST_STD_VER >= 11
  test_string<std::basic_string<char, std::char_traits<char>, min_allocator<char> > >();
  test_string<std::basic_string<char, std::char_traits<char>, safe_allocator<char> > >();
#endif

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 20
  static_assert(test());
#endif

  return 0;
}
