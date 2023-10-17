//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

//    template<class charT, class traits>
//        explicit bitset(
//            const basic_string_view<charT,traits>& str,
//            typename basic_string_view<charT,traits>::size_type pos = 0,
//            typename basic_string_view<charT,traits>::size_type n = basic_string_view<charT,traits>::npos,
//            charT zero = charT('0'), charT one = charT('1'));

#include <algorithm> // for 'min' and 'max'
#include <bitset>
#include <cassert>
#include <stdexcept> // for 'invalid_argument'
#include <string_view>
#include <type_traits>

#include "test_macros.h"

template <std::size_t N>
constexpr void test_string_ctor() {
#ifndef TEST_HAS_NO_EXCEPTIONS
  if (!TEST_IS_CONSTANT_EVALUATED) {
    try {
      std::string_view s("xxx1010101010xxxx");
      std::bitset<N> v(s, s.size() + 1);
      assert(false);
    } catch (std::out_of_range&) {
    }
    try {
      std::string_view s("xxx1010101010xxxx");
      std::bitset<N> v(s, s.size() + 1, 10);
      assert(false);
    } catch (std::out_of_range&) {
    }
    try {
      std::string_view s("xxx1010101010xxxx");
      std::bitset<N> v(s);
      assert(false);
    } catch (std::invalid_argument&) {
    }
    try {
      std::string_view s("xxx1010101010xxxx");
      std::bitset<N> v(s, 2);
      assert(false);
    } catch (std::invalid_argument&) {
    }
    try {
      std::string_view s("xxx1010101010xxxx");
      std::bitset<N> v(s, 2, 10);
      assert(false);
    } catch (std::invalid_argument&) {
    }
    try {
      std::string_view s("xxxbababababaxxxx");
      std::bitset<N> v(s, 2, 10, 'a', 'b');
      assert(false);
    } catch (std::invalid_argument&) {
    }
  }
#endif // TEST_HAS_NO_EXCEPTIONS

  static_assert(!std::is_convertible_v<std::string_view, std::bitset<N>>);
  static_assert(std::is_constructible_v<std::bitset<N>, std::string_view>);
  {
    std::string_view s("1010101010");
    std::bitset<N> v(s);
    std::size_t M = std::min<std::size_t>(v.size(), 10);
    for (std::size_t i = 0; i < M; ++i)
      assert(v[i] == (s[M - 1 - i] == '1'));
    for (std::size_t i = 10; i < v.size(); ++i)
      assert(v[i] == false);
  }
  {
    std::string_view s("xxx1010101010");
    std::bitset<N> v(s, 3);
    std::size_t M = std::min<std::size_t>(v.size(), 10);
    for (std::size_t i = 0; i < M; ++i)
      assert(v[i] == (s[3 + M - 1 - i] == '1'));
    for (std::size_t i = 10; i < v.size(); ++i)
      assert(v[i] == false);
  }
  {
    std::string_view s("xxx1010101010xxxx");
    std::bitset<N> v(s, 3, 10);
    std::size_t M = std::min<std::size_t>(v.size(), 10);
    for (std::size_t i = 0; i < M; ++i)
      assert(v[i] == (s[3 + M - 1 - i] == '1'));
    for (std::size_t i = 10; i < v.size(); ++i)
      assert(v[i] == false);
  }
  {
    std::string_view s("xxx1a1a1a1a1axxxx");
    std::bitset<N> v(s, 3, 10, 'a');
    std::size_t M = std::min<std::size_t>(v.size(), 10);
    for (std::size_t i = 0; i < M; ++i)
      assert(v[i] == (s[3 + M - 1 - i] == '1'));
    for (std::size_t i = 10; i < v.size(); ++i)
      assert(v[i] == false);
  }
  {
    std::string_view s("xxxbababababaxxxx");
    std::bitset<N> v(s, 3, 10, 'a', 'b');
    std::size_t M = std::min<std::size_t>(v.size(), 10);
    for (std::size_t i = 0; i < M; ++i)
      assert(v[i] == (s[3 + M - 1 - i] == 'b'));
    for (std::size_t i = 10; i < v.size(); ++i)
      assert(v[i] == false);
  }
}

struct Nonsense {
  virtual ~Nonsense() {}
};

constexpr void test_for_non_eager_instantiation() {
  // Ensure we don't accidentally instantiate `std::basic_string_view<Nonsense>`
  // since it may not be well formed and can cause an error in the
  // non-immediate context.
  static_assert(!std::is_constructible<std::bitset<3>, Nonsense*>::value, "");
  static_assert(!std::is_constructible<std::bitset<3>, Nonsense*, std::size_t, Nonsense&, Nonsense&>::value, "");
}

constexpr bool test() {
  test_string_ctor<0>();
  test_string_ctor<1>();
  test_string_ctor<31>();
  test_string_ctor<32>();
  test_string_ctor<33>();
  test_string_ctor<63>();
  test_string_ctor<64>();
  test_string_ctor<65>();
  test_string_ctor<1000>();
  test_for_non_eager_instantiation();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
