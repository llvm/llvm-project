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
//   operator+(const basic_string<charT,traits,Allocator>& lhs, charT rhs); // constexpr since C++20

// template<class charT, class traits, class Allocator>
//   basic_string<charT,traits,Allocator>&&
//   operator+(basic_string<charT,traits,Allocator>&& lhs, charT rhs); // constexpr since C++20

#include <cassert>
#include <string>
#include <utility>

#include "asan_testing.h"
#include "min_allocator.h"
#include "test_macros.h"

template <class S>
TEST_CONSTEXPR_CXX20 void test_string() {
  const char* test_data[] = {"", "12345", "1234567890", "12345678901234567890"};
  const char* results[]   = {"a", "12345a", "1234567890a", "12345678901234567890a"};

  for (size_t i = 0; i != 4; ++i) {
    { // operator+(const string&, value_type);
      const S str(test_data[i]);
      assert(str + 'a' == results[i]);
      LIBCPP_ASSERT(is_string_asan_correct(str + 'a'));
    }
#if TEST_STD_VER >= 11
    { // operator+(string&&, value_type);
      S str(test_data[i]);
      assert(std::move(str) + 'a' == results[i]);
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

  { // check that growing to max_size() works
    using string_type = std::basic_string<char, std::char_traits<char>, tiny_size_allocator<29, char> >;
    string_type str;
    str.resize(str.max_size() - 1);
    string_type result = str + 'a';

    assert(result.size() == result.max_size());
    assert(result.back() == 'a');
    assert(result.capacity() <= result.get_allocator().max_size());
  }

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 20
  static_assert(test());
#endif

  return 0;
}
