//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// template <class charT>
//     explicit bitset(const charT* str,
//                     typename basic_string_view<charT>::size_type n = basic_string_view<charT>::npos, // s/string/string_view since C++26
//                     charT zero = charT('0'), charT one = charT('1')); // constexpr since C++23

#include <bitset>
#include <algorithm> // for 'min' and 'max'
#include <cassert>
#include <stdexcept> // for 'invalid_argument'
#include <type_traits>

#include "test_macros.h"

TEST_MSVC_DIAGNOSTIC_IGNORED(6294) // Ill-defined for-loop:  initial condition does not satisfy test.  Loop body not executed.

template <std::size_t N>
TEST_CONSTEXPR_CXX23 void test_char_pointer_ctor()
{
#ifndef TEST_HAS_NO_EXCEPTIONS
  if (!TEST_IS_CONSTANT_EVALUATED) {
    try {
      std::bitset<N> v("xxx1010101010xxxx");
      assert(false);
    }
    catch (std::invalid_argument&) {}
  }
#endif

  static_assert(!std::is_convertible<const char*, std::bitset<N> >::value, "");
  static_assert(std::is_constructible<std::bitset<N>, const char*>::value, "");
  {
    const char s[] = "1010101010";
    std::bitset<N> v(s);
    std::size_t M = std::min<std::size_t>(v.size(), 10);
    for (std::size_t i = 0; i < M; ++i)
        assert(v[i] == (s[M - 1 - i] == '1'));
    for (std::size_t i = 10; i < v.size(); ++i)
        assert(v[i] == false);
  }
  {
    const char s[] = "1010101010";
    std::bitset<N> v(s, 10);
    std::size_t M = std::min<std::size_t>(v.size(), 10);
    for (std::size_t i = 0; i < M; ++i)
        assert(v[i] == (s[M - 1 - i] == '1'));
    for (std::size_t i = 10; i < v.size(); ++i)
        assert(v[i] == false);
  }
  {
    const char s[] = "1a1a1a1a1a";
    std::bitset<N> v(s, 10, 'a');
    std::size_t M = std::min<std::size_t>(v.size(), 10);
    for (std::size_t i = 0; i < M; ++i)
        assert(v[i] == (s[M - 1 - i] == '1'));
    for (std::size_t i = 10; i < v.size(); ++i)
        assert(v[i] == false);
  }
  {
    const char s[] = "bababababa";
    std::bitset<N> v(s, 10, 'a', 'b');
    std::size_t M = std::min<std::size_t>(v.size(), 10);
    for (std::size_t i = 0; i < M; ++i)
        assert(v[i] == (s[M - 1 - i] == 'b'));
    for (std::size_t i = 10; i < v.size(); ++i)
        assert(v[i] == false);
  }
}

TEST_CONSTEXPR_CXX23 bool test() {
  test_char_pointer_ctor<0>();
  test_char_pointer_ctor<1>();
  test_char_pointer_ctor<31>();
  test_char_pointer_ctor<32>();
  test_char_pointer_ctor<33>();
  test_char_pointer_ctor<63>();
  test_char_pointer_ctor<64>();
  test_char_pointer_ctor<65>();
  test_char_pointer_ctor<1000>();

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER > 20
  static_assert(test());
#endif

  return 0;
}
