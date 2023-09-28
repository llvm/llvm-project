//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// we get this comparison "for free" because the string implicitly converts to the string_view

#include <string>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

template <class S, class SV>
TEST_CONSTEXPR_CXX20 void test(SV lhs, const S& rhs, bool x) {
  assert((lhs > rhs) == x);
}

template <class CharT, template <class> class Alloc>
TEST_CONSTEXPR_CXX20 void test_string() {
  using S  = std::basic_string<CharT, std::char_traits<CharT>, Alloc<CharT> >;
  using SV = std::basic_string_view<CharT, std::char_traits<CharT> >;
  test(SV(""), S(""), false);
  test(SV(""), S("abcde"), false);
  test(SV(""), S("abcdefghij"), false);
  test(SV(""), S("abcdefghijklmnopqrst"), false);
  test(SV("abcde"), S(""), true);
  test(SV("abcde"), S("abcde"), false);
  test(SV("abcde"), S("abcdefghij"), false);
  test(SV("abcde"), S("abcdefghijklmnopqrst"), false);
  test(SV("abcdefghij"), S(""), true);
  test(SV("abcdefghij"), S("abcde"), true);
  test(SV("abcdefghij"), S("abcdefghij"), false);
  test(SV("abcdefghij"), S("abcdefghijklmnopqrst"), false);
  test(SV("abcdefghijklmnopqrst"), S(""), true);
  test(SV("abcdefghijklmnopqrst"), S("abcde"), true);
  test(SV("abcdefghijklmnopqrst"), S("abcdefghij"), true);
  test(SV("abcdefghijklmnopqrst"), S("abcdefghijklmnopqrst"), false);
}

TEST_CONSTEXPR_CXX20 bool test() {
  test_string<char, std::allocator>();
#if TEST_STD_VER >= 11
  test_string<char, min_allocator>();
#endif

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER > 17
  static_assert(test());
#endif

  return 0;
}
