//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <bitset>

// template <class charT>
//     explicit bitset(const charT* str,
//                     typename basic_string_view<charT>::size_type n = basic_string_view<charT>::npos,
//                     charT zero = charT('0'), charT one = charT('1'));
//
// Constraints:
//   - is_array_v<charT> is false,
//   - is_trivially_copyable_v<charT> is true,
//   - is_standard_layout_v<charT> is true, and
//   - is_trivially_default_constructible_v<charT> is true.

// The frozen C++03 headers don't implement LWG4294.
// XFAIL: FROZEN-CXX03-HEADERS-FIXME

#include <bitset>
#include <cstddef>
#include <type_traits>

#include "test_macros.h"

struct NonTrivialDestructor {
  ~NonTrivialDestructor() {}
};

struct NonTrivialDefaultCtor {
  NonTrivialDefaultCtor() {}
};

struct NonTriviallyCopyable {
  NonTriviallyCopyable& operator=(const NonTriviallyCopyable&) { return *this; }
};

struct NonStandardLayout {
  int a;

protected:
  int b;
};

template <std::size_t N>
void test() {
  // Character types satisfying the constraints.
  static_assert(std::is_constructible<std::bitset<N>, const char*>::value, "");
  static_assert(std::is_constructible<std::bitset<N>, char*>::value, "");
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  static_assert(std::is_constructible<std::bitset<N>, const wchar_t*>::value, "");
#endif

  // Array types.
  static_assert(!std::is_constructible<std::bitset<N>, const char (*)[2]>::value, "");
  static_assert(!std::is_constructible<std::bitset<N>, char (*)[2]>::value, "");
  static_assert(!std::is_constructible<std::bitset<N>, const int (*)[3]>::value, "");
  static_assert(!std::is_constructible<std::bitset<N>, const char (*)[2][2]>::value, "");

  // Class types that are not trivially copyable, standard-layout, or trivially default constructible.
  static_assert(!std::is_constructible<std::bitset<N>, NonTrivialDestructor*>::value, "");
  static_assert(!std::is_constructible<std::bitset<N>, const NonTrivialDefaultCtor*>::value, "");
  static_assert(!std::is_constructible<std::bitset<N>, const NonTriviallyCopyable*>::value, "");
  static_assert(!std::is_constructible<std::bitset<N>, const NonStandardLayout*>::value, "");
}

void test() {
  test<0>();
  test<1>();
  test<32>();
  test<64>();
  test<65>();
}
