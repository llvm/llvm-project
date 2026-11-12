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
//   operator+(const charT* lhs, const basic_string<charT,traits,Allocator>& rhs); // constexpr since C++20

// template<class charT, class traits, class Allocator>
//   basic_string<charT,traits,Allocator>&&
//   operator+(const charT* lhs, basic_string<charT,traits,Allocator>&& rhs); // constexpr since C++20

#include <cassert>
#include <string>
#include <utility>

#include "test_macros.h"
#include "min_allocator.h"

template <class S>
TEST_CONSTEXPR_CXX20 void test_string() {
  const char* test_data[2][4] = {
      {"", "abcde", "abcdefghij", "abcdefghijklmnopqrst"}, {"", "12345", "1234567890", "12345678901234567890"}};

  const char* results[4][4] = {
      {"", "12345", "1234567890", "12345678901234567890"},
      {"abcde", "abcde12345", "abcde1234567890", "abcde12345678901234567890"},
      {"abcdefghij", "abcdefghij12345", "abcdefghij1234567890", "abcdefghij12345678901234567890"},
      {"abcdefghijklmnopqrst",
       "abcdefghijklmnopqrst12345",
       "abcdefghijklmnopqrst1234567890",
       "abcdefghijklmnopqrst12345678901234567890"}};

  for (size_t i = 0; i != 4; ++i) {
    for (size_t k = 0; k != 4; ++k) {
      { // operator+(const value_type*, const string&);
        const char* lhs = test_data[0][i];
        const S rhs(test_data[1][k]);
        assert(lhs + rhs == results[i][k]);
      }
#if TEST_STD_VER >= 11
      { // operator+(const value_type*, string&&);
        const char* lhs = test_data[0][i];
        S rhs(test_data[1][k]);
        assert(lhs + std::move(rhs) == results[i][k]);
      }
#endif
    }
  }
}

TEST_CONSTEXPR_CXX20 bool test() {
  test_string<std::string>();
#if TEST_STD_VER >= 11
  test_string<std::basic_string<char, std::char_traits<char>, min_allocator<char>>>();
#endif

  { // check that growing to max_size() works
    using string_type = std::basic_string<char, std::char_traits<char>, tiny_size_allocator<29, char> >;
    string_type str;
    str.resize(str.max_size() - 1);
    string_type result = "a" + str;

    assert(result.size() == result.max_size());
    assert(result.front() == 'a');
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
