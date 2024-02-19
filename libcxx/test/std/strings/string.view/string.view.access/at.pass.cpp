//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string_view>

// constexpr const _CharT& at(size_type _pos) const;

#include <string_view>
#include <stdexcept>
#include <cassert>

#include "test_macros.h"

template <typename CharT>
void test(const CharT* s, std::size_t len) {
  std::basic_string_view<CharT> sv(s, len);
  assert(sv.length() == len);
  for (std::size_t i = 0; i < len; ++i) {
    assert(sv.at(i) == s[i]);
    assert(&sv.at(i) == s + i);
  }

#ifndef TEST_HAS_NO_EXCEPTIONS
  try {
    (void)sv.at(len);
  } catch (const std::out_of_range&) {
    return;
  }
  assert(false);
#endif
}

int main(int, char**) {
  test("ABCDE", 5);
  test("a", 1);

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test(L"ABCDE", 5);
  test(L"a", 1);
#endif

#if TEST_STD_VER >= 11
  test(u"ABCDE", 5);
  test(u"a", 1);

  test(U"ABCDE", 5);
  test(U"a", 1);
#endif

#if TEST_STD_VER >= 11
  {
    constexpr std::basic_string_view<char> sv("ABC", 2);
    static_assert(sv.length() == 2, "");
    static_assert(sv.at(0) == 'A', "");
    static_assert(sv.at(1) == 'B', "");
  }
#endif

  return 0;
}
