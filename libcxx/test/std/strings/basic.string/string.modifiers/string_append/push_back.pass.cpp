//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// void push_back(charT c) // constexpr since C++20

#include <string>
#include <algorithm>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"
#include "asan_testing.h"

struct VeryLarge {
  long long a;
  char b;
};

template <>
struct std::char_traits<VeryLarge> {
  using char_type  = VeryLarge;
  using int_type   = int;
  using off_type   = streamoff;
  using pos_type   = streampos;
  using state_type = mbstate_t;

  static TEST_CONSTEXPR_CXX20 void assign(char_type& c1, const char_type& c2) { c1 = c2; }
  static bool eq(char_type c1, char_type c2);
  static bool lt(char_type c1, char_type c2);

  static int compare(const char_type* s1, const char_type* s2, std::size_t n);
  static std::size_t length(const char_type* s);
  static const char_type* find(const char_type* s, std::size_t n, const char_type& a);
  static char_type* move(char_type* s1, const char_type* s2, std::size_t n);
  static TEST_CONSTEXPR_CXX20 char_type* copy(char_type* s1, const char_type* s2, std::size_t n) {
    std::copy_n(s2, n, s1);
    return s1;
  }
  static TEST_CONSTEXPR_CXX20 char_type* assign(char_type* s, std::size_t n, char_type a) {
    std::fill_n(s, n, a);
    return s;
  }

  static int_type not_eof(int_type c);
  static char_type to_char_type(int_type c);
  static int_type to_int_type(char_type c);
  static bool eq_int_type(int_type c1, int_type c2);
  static int_type eof();
};

template <class S>
TEST_CONSTEXPR_CXX20 void test(S s, typename S::value_type c, S expected) {
  s.push_back(c);
  LIBCPP_ASSERT(s.__invariants());
  assert(s == expected);
  LIBCPP_ASSERT(is_string_asan_correct(s));
}

template <class S>
TEST_CONSTEXPR_CXX20 void test_string() {
  test(S(), 'a', S(1, 'a'));
  test(S("12345"), 'a', S("12345a"));
  test(S("12345678901234567890"), 'a', S("12345678901234567890a"));
  test(S("123abcabcdefghabcdefgh"), 'a', S("123abcabcdefghabcdefgha"));
  test(S("123abcabcdefghabcdefgha"), 'b', S("123abcabcdefghabcdefghab"));
  test(S("123abcabcdefghabcdefghab"), 'c', S("123abcabcdefghabcdefghabc"));
  test(S("123abcabcdefghabcdefghabc"), 'd', S("123abcabcdefghabcdefghabcd"));
}

TEST_CONSTEXPR_CXX20 bool test() {
  test_string<std::string>();
#if TEST_STD_VER >= 11
  test_string<std::basic_string<char, std::char_traits<char>, min_allocator<char> > >();
  test_string<std::basic_string<char, std::char_traits<char>, safe_allocator<char> > >();
#endif
  {
    // https://llvm.org/PR31454
    std::basic_string<VeryLarge> s;
    VeryLarge vl = {};
    s.push_back(vl);
    s.push_back(vl);
    s.push_back(vl);
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
