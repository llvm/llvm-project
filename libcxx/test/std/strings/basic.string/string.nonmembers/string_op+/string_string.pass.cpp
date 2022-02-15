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
//   operator+(const basic_string<charT,traits,Allocator>& lhs,
//             const basic_string<charT,traits,Allocator>& rhs);

// template<class charT, class traits, class Allocator>
//   basic_string<charT,traits,Allocator>&&
//   operator+(const basic_string<charT,traits,Allocator>&& lhs,
//             const basic_string<charT,traits,Allocator>& rhs);

// template<class charT, class traits, class Allocator>
//   basic_string<charT,traits,Allocator>&&
//   operator+(const basic_string<charT,traits,Allocator>& lhs,
//             const basic_string<charT,traits,Allocator>&& rhs);

// template<class charT, class traits, class Allocator>
//   basic_string<charT,traits,Allocator>&&
//   operator+(const basic_string<charT,traits,Allocator>&& lhs,
//             const basic_string<charT,traits,Allocator>&& rhs);

#include <string>
#include <utility>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

template <class S>
TEST_CONSTEXPR_CXX20 void test0(const S& lhs, const S& rhs, const S& x) {
  assert(lhs + rhs == x);
}

#if TEST_STD_VER >= 11
template <class S>
TEST_CONSTEXPR_CXX20 void test1(S&& lhs, const S& rhs, const S& x) {
  assert(move(lhs) + rhs == x);
}

template <class S>
TEST_CONSTEXPR_CXX20 void test2(const S& lhs, S&& rhs, const S& x) {
  assert(lhs + move(rhs) == x);
}

template <class S>
TEST_CONSTEXPR_CXX20 void test3(S&& lhs, S&& rhs, const S& x) {
  assert(move(lhs) + move(rhs) == x);
}
#endif

bool test() {
  {
    typedef std::string S;
    test0(S(""), S(""), S(""));
    test0(S(""), S("12345"), S("12345"));
    test0(S(""), S("1234567890"), S("1234567890"));
    test0(S(""), S("12345678901234567890"), S("12345678901234567890"));
    test0(S("abcde"), S(""), S("abcde"));
    test0(S("abcde"), S("12345"), S("abcde12345"));
    test0(S("abcde"), S("1234567890"), S("abcde1234567890"));
    test0(S("abcde"), S("12345678901234567890"),
          S("abcde12345678901234567890"));
    test0(S("abcdefghij"), S(""), S("abcdefghij"));
    test0(S("abcdefghij"), S("12345"), S("abcdefghij12345"));
    test0(S("abcdefghij"), S("1234567890"), S("abcdefghij1234567890"));
    test0(S("abcdefghij"), S("12345678901234567890"),
          S("abcdefghij12345678901234567890"));
    test0(S("abcdefghijklmnopqrst"), S(""), S("abcdefghijklmnopqrst"));
    test0(S("abcdefghijklmnopqrst"), S("12345"),
          S("abcdefghijklmnopqrst12345"));
    test0(S("abcdefghijklmnopqrst"), S("1234567890"),
          S("abcdefghijklmnopqrst1234567890"));
    test0(S("abcdefghijklmnopqrst"), S("12345678901234567890"),
          S("abcdefghijklmnopqrst12345678901234567890"));
  }
#if TEST_STD_VER >= 11
  {
    typedef std::string S;
    test1(S(""), S(""), S(""));
    test1(S(""), S("12345"), S("12345"));
    test1(S(""), S("1234567890"), S("1234567890"));
    test1(S(""), S("12345678901234567890"), S("12345678901234567890"));
    test1(S("abcde"), S(""), S("abcde"));
    test1(S("abcde"), S("12345"), S("abcde12345"));
    test1(S("abcde"), S("1234567890"), S("abcde1234567890"));
    test1(S("abcde"), S("12345678901234567890"),
          S("abcde12345678901234567890"));
    test1(S("abcdefghij"), S(""), S("abcdefghij"));
    test1(S("abcdefghij"), S("12345"), S("abcdefghij12345"));
    test1(S("abcdefghij"), S("1234567890"), S("abcdefghij1234567890"));
    test1(S("abcdefghij"), S("12345678901234567890"),
          S("abcdefghij12345678901234567890"));
    test1(S("abcdefghijklmnopqrst"), S(""), S("abcdefghijklmnopqrst"));
    test1(S("abcdefghijklmnopqrst"), S("12345"),
          S("abcdefghijklmnopqrst12345"));
    test1(S("abcdefghijklmnopqrst"), S("1234567890"),
          S("abcdefghijklmnopqrst1234567890"));
    test1(S("abcdefghijklmnopqrst"), S("12345678901234567890"),
          S("abcdefghijklmnopqrst12345678901234567890"));

    test2(S(""), S(""), S(""));
    test2(S(""), S("12345"), S("12345"));
    test2(S(""), S("1234567890"), S("1234567890"));
    test2(S(""), S("12345678901234567890"), S("12345678901234567890"));
    test2(S("abcde"), S(""), S("abcde"));
    test2(S("abcde"), S("12345"), S("abcde12345"));
    test2(S("abcde"), S("1234567890"), S("abcde1234567890"));
    test2(S("abcde"), S("12345678901234567890"),
          S("abcde12345678901234567890"));
    test2(S("abcdefghij"), S(""), S("abcdefghij"));
    test2(S("abcdefghij"), S("12345"), S("abcdefghij12345"));
    test2(S("abcdefghij"), S("1234567890"), S("abcdefghij1234567890"));
    test2(S("abcdefghij"), S("12345678901234567890"),
          S("abcdefghij12345678901234567890"));
    test2(S("abcdefghijklmnopqrst"), S(""), S("abcdefghijklmnopqrst"));
    test2(S("abcdefghijklmnopqrst"), S("12345"),
          S("abcdefghijklmnopqrst12345"));
    test2(S("abcdefghijklmnopqrst"), S("1234567890"),
          S("abcdefghijklmnopqrst1234567890"));
    test2(S("abcdefghijklmnopqrst"), S("12345678901234567890"),
          S("abcdefghijklmnopqrst12345678901234567890"));

    test3(S(""), S(""), S(""));
    test3(S(""), S("12345"), S("12345"));
    test3(S(""), S("1234567890"), S("1234567890"));
    test3(S(""), S("12345678901234567890"), S("12345678901234567890"));
    test3(S("abcde"), S(""), S("abcde"));
    test3(S("abcde"), S("12345"), S("abcde12345"));
    test3(S("abcde"), S("1234567890"), S("abcde1234567890"));
    test3(S("abcde"), S("12345678901234567890"),
          S("abcde12345678901234567890"));
    test3(S("abcdefghij"), S(""), S("abcdefghij"));
    test3(S("abcdefghij"), S("12345"), S("abcdefghij12345"));
    test3(S("abcdefghij"), S("1234567890"), S("abcdefghij1234567890"));
    test3(S("abcdefghij"), S("12345678901234567890"),
          S("abcdefghij12345678901234567890"));
    test3(S("abcdefghijklmnopqrst"), S(""), S("abcdefghijklmnopqrst"));
    test3(S("abcdefghijklmnopqrst"), S("12345"),
          S("abcdefghijklmnopqrst12345"));
    test3(S("abcdefghijklmnopqrst"), S("1234567890"),
          S("abcdefghijklmnopqrst1234567890"));
    test3(S("abcdefghijklmnopqrst"), S("12345678901234567890"),
          S("abcdefghijklmnopqrst12345678901234567890"));
  }
  {
    typedef std::basic_string<char, std::char_traits<char>,
                              min_allocator<char> >
        S;
    test0(S(""), S(""), S(""));
    test0(S(""), S("12345"), S("12345"));
    test0(S(""), S("1234567890"), S("1234567890"));
    test0(S(""), S("12345678901234567890"), S("12345678901234567890"));
    test0(S("abcde"), S(""), S("abcde"));
    test0(S("abcde"), S("12345"), S("abcde12345"));
    test0(S("abcde"), S("1234567890"), S("abcde1234567890"));
    test0(S("abcde"), S("12345678901234567890"),
          S("abcde12345678901234567890"));
    test0(S("abcdefghij"), S(""), S("abcdefghij"));
    test0(S("abcdefghij"), S("12345"), S("abcdefghij12345"));
    test0(S("abcdefghij"), S("1234567890"), S("abcdefghij1234567890"));
    test0(S("abcdefghij"), S("12345678901234567890"),
          S("abcdefghij12345678901234567890"));
    test0(S("abcdefghijklmnopqrst"), S(""), S("abcdefghijklmnopqrst"));
    test0(S("abcdefghijklmnopqrst"), S("12345"),
          S("abcdefghijklmnopqrst12345"));
    test0(S("abcdefghijklmnopqrst"), S("1234567890"),
          S("abcdefghijklmnopqrst1234567890"));
    test0(S("abcdefghijklmnopqrst"), S("12345678901234567890"),
          S("abcdefghijklmnopqrst12345678901234567890"));

    test1(S(""), S(""), S(""));
    test1(S(""), S("12345"), S("12345"));
    test1(S(""), S("1234567890"), S("1234567890"));
    test1(S(""), S("12345678901234567890"), S("12345678901234567890"));
    test1(S("abcde"), S(""), S("abcde"));
    test1(S("abcde"), S("12345"), S("abcde12345"));
    test1(S("abcde"), S("1234567890"), S("abcde1234567890"));
    test1(S("abcde"), S("12345678901234567890"),
          S("abcde12345678901234567890"));
    test1(S("abcdefghij"), S(""), S("abcdefghij"));
    test1(S("abcdefghij"), S("12345"), S("abcdefghij12345"));
    test1(S("abcdefghij"), S("1234567890"), S("abcdefghij1234567890"));
    test1(S("abcdefghij"), S("12345678901234567890"),
          S("abcdefghij12345678901234567890"));
    test1(S("abcdefghijklmnopqrst"), S(""), S("abcdefghijklmnopqrst"));
    test1(S("abcdefghijklmnopqrst"), S("12345"),
          S("abcdefghijklmnopqrst12345"));
    test1(S("abcdefghijklmnopqrst"), S("1234567890"),
          S("abcdefghijklmnopqrst1234567890"));
    test1(S("abcdefghijklmnopqrst"), S("12345678901234567890"),
          S("abcdefghijklmnopqrst12345678901234567890"));

    test2(S(""), S(""), S(""));
    test2(S(""), S("12345"), S("12345"));
    test2(S(""), S("1234567890"), S("1234567890"));
    test2(S(""), S("12345678901234567890"), S("12345678901234567890"));
    test2(S("abcde"), S(""), S("abcde"));
    test2(S("abcde"), S("12345"), S("abcde12345"));
    test2(S("abcde"), S("1234567890"), S("abcde1234567890"));
    test2(S("abcde"), S("12345678901234567890"),
          S("abcde12345678901234567890"));
    test2(S("abcdefghij"), S(""), S("abcdefghij"));
    test2(S("abcdefghij"), S("12345"), S("abcdefghij12345"));
    test2(S("abcdefghij"), S("1234567890"), S("abcdefghij1234567890"));
    test2(S("abcdefghij"), S("12345678901234567890"),
          S("abcdefghij12345678901234567890"));
    test2(S("abcdefghijklmnopqrst"), S(""), S("abcdefghijklmnopqrst"));
    test2(S("abcdefghijklmnopqrst"), S("12345"),
          S("abcdefghijklmnopqrst12345"));
    test2(S("abcdefghijklmnopqrst"), S("1234567890"),
          S("abcdefghijklmnopqrst1234567890"));
    test2(S("abcdefghijklmnopqrst"), S("12345678901234567890"),
          S("abcdefghijklmnopqrst12345678901234567890"));

    test3(S(""), S(""), S(""));
    test3(S(""), S("12345"), S("12345"));
    test3(S(""), S("1234567890"), S("1234567890"));
    test3(S(""), S("12345678901234567890"), S("12345678901234567890"));
    test3(S("abcde"), S(""), S("abcde"));
    test3(S("abcde"), S("12345"), S("abcde12345"));
    test3(S("abcde"), S("1234567890"), S("abcde1234567890"));
    test3(S("abcde"), S("12345678901234567890"),
          S("abcde12345678901234567890"));
    test3(S("abcdefghij"), S(""), S("abcdefghij"));
    test3(S("abcdefghij"), S("12345"), S("abcdefghij12345"));
    test3(S("abcdefghij"), S("1234567890"), S("abcdefghij1234567890"));
    test3(S("abcdefghij"), S("12345678901234567890"),
          S("abcdefghij12345678901234567890"));
    test3(S("abcdefghijklmnopqrst"), S(""), S("abcdefghijklmnopqrst"));
    test3(S("abcdefghijklmnopqrst"), S("12345"),
          S("abcdefghijklmnopqrst12345"));
    test3(S("abcdefghijklmnopqrst"), S("1234567890"),
          S("abcdefghijklmnopqrst1234567890"));
    test3(S("abcdefghijklmnopqrst"), S("12345678901234567890"),
          S("abcdefghijklmnopqrst12345678901234567890"));
  }
#endif // TEST_STD_VER >= 11

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER > 17
  // static_assert(test());
#endif

  return 0;
}
