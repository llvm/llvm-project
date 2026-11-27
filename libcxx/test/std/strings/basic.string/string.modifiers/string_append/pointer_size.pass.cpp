//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// basic_string<charT,traits,Allocator>&
//   append(const charT* s, size_type n); // constexpr since C++20

#include <string>
#include <stdexcept>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"
#include "asan_testing.h"

template <class S>
TEST_CONSTEXPR_CXX20 void test(S s, const typename S::value_type* str, typename S::size_type n, S expected) {
  s.append(str, n);
  LIBCPP_ASSERT(s.__invariants());
  assert(s == expected);
  LIBCPP_ASSERT(is_string_asan_correct(s));
}

template <class S>
TEST_CONSTEXPR_CXX20 void test_string() {
  test(S(), "", 0, S());
  test(S(), "12345", 3, S("123"));
  test(S(), "12345", 4, S("1234"));
  test(S(), "12345678901234567890", 0, S());
  test(S(), "12345678901234567890", 1, S("1"));
  test(S(), "12345678901234567890", 3, S("123"));
  test(S(), "12345678901234567890", 20, S("12345678901234567890"));
  test(S(), "1234567890123456789012345678901234567890", 40, S("1234567890123456789012345678901234567890"));

  test(S("12345"), "", 0, S("12345"));
  test(S("12345"), "12345", 5, S("1234512345"));
  test(S("12345"), "1234567890", 10, S("123451234567890"));

  test(S("12345678901234567890"), "", 0, S("12345678901234567890"));
  test(S("12345678901234567890"), "12345", 5, S("1234567890123456789012345"));
  test(S("12345678901234567890"), "12345678901234567890", 20, S("1234567890123456789012345678901234567890"));

  // Starting from long string (no SSO)
  test(S("1234567890123456789012345678901234567890"), "", 0, S("1234567890123456789012345678901234567890"));
  test(S("1234567890123456789012345678901234567890"), "a", 1, S("1234567890123456789012345678901234567890a"));
  test(S("1234567890123456789012345678901234567890"),
       "aaaaaaaaaa",
       10,
       S("1234567890123456789012345678901234567890aaaaaaaaaa"));
  test(S("1234567890123456789012345678901234567890"),
       "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
       "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
       "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
       300,
       S("1234567890123456789012345678901234567890aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
         "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
         "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
         "aaaaaaaaaaaaa"));
}

TEST_CONSTEXPR_CXX20 bool test() {
  test_string<std::string>();
#if TEST_STD_VER >= 11
  test_string<std::basic_string<char, std::char_traits<char>, min_allocator<char>>>();
  test_string<std::basic_string<char, std::char_traits<char>, safe_allocator<char>>>();
#endif

  { // test appending to self
    typedef std::string S;
    S s_short = "123/";
    S s_long  = "Lorem ipsum dolor sit amet, consectetur/";

    s_short.append(s_short.data(), s_short.size());
    assert(s_short == "123/123/");
    s_short.append(s_short.data(), s_short.size());
    assert(s_short == "123/123/123/123/");
    s_short.append(s_short.data(), s_short.size());
    assert(s_short == "123/123/123/123/123/123/123/123/");

    s_long.append(s_long.data(), s_long.size());
    assert(s_long == "Lorem ipsum dolor sit amet, consectetur/Lorem ipsum dolor sit amet, consectetur/");
  }

  { // check that growing to max_size() works
    using string_type = std::basic_string<char, std::char_traits<char>, tiny_size_allocator<29, char> >;
    string_type str;
    auto max_size = str.max_size();
    str.resize(max_size / 2 + max_size % 2);
    str.append(str.c_str(), max_size / 2);
    assert(str.capacity() >= str.size());
    assert(str.size() == str.max_size());
  }

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER > 17
  static_assert(test());
#endif

  return 0;
}
