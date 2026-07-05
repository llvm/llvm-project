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
TEST_CONSTEXPR_CXX20 void test(const S& lhs, SV rhs, bool x) {
  assert((lhs > rhs) == x);
}

template <class CharT, template <class> class Alloc>
TEST_CONSTEXPR_CXX20 void test_string() {
  using S  = std::basic_string<CharT, std::char_traits<CharT>, Alloc<CharT> >;
  using SV = std::basic_string_view<CharT, std::char_traits<CharT> >;
  test(S(""), SV(""), false);
  test(S(""), SV("abcde"), false);
  test(S(""), SV("abcdefghij"), false);
  test(S(""), SV("abcdefghijklmnopqrst"), false);
  test(S("abcde"), SV(""), true);
  test(S("abcde"), SV("abcde"), false);
  test(S("abcde"), SV("abcdefghij"), false);
  test(S("abcde"), SV("abcdefghijklmnopqrst"), false);
  test(S("abcdefghij"), SV(""), true);
  test(S("abcdefghij"), SV("abcde"), true);
  test(S("abcdefghij"), SV("abcdefghij"), false);
  test(S("abcdefghij"), SV("abcdefghijklmnopqrst"), false);
  test(S("abcdefghijklmnopqrst"), SV(""), true);
  test(S("abcdefghijklmnopqrst"), SV("abcde"), true);
  test(S("abcdefghijklmnopqrst"), SV("abcdefghij"), true);
  test(S("abcdefghijklmnopqrst"), SV("abcdefghijklmnopqrst"), false);
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
