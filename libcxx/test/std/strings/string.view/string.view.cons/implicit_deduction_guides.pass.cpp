//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <string_view>

// Test that the constructors offered by std::basic_string_view are formulated
// so they're compatible with implicit deduction guides.

#include <string_view>
#include <cassert>

#include "test_macros.h"
#include "constexpr_char_traits.h"

// Overloads
// ---------------
// (1)  basic_string_view() - NOT TESTED
// (2)  basic_string_view(const basic_string_view&)
// (3)  basic_string_view(const CharT*, size_type)
// (4)  basic_string_view(const CharT*)
int main(int, char**) {
  {
    // Testing (1)
    // Nothing to do. Cannot deduce without any arguments.
  }
  { // Testing (2)
    const std::string_view sin("abc");
    std::basic_string_view s(sin);
    ASSERT_SAME_TYPE(decltype(s), std::string_view);
    assert(s == "abc");

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    using WSV = std::basic_string_view<wchar_t, constexpr_char_traits<wchar_t>>;
    const WSV win(L"abcdef");
    std::basic_string_view w(win);
    ASSERT_SAME_TYPE(decltype(w), WSV);
    assert(w == L"abcdef");
#endif
  }
  { // Testing (3)
    std::basic_string_view s("abc", 2);
    ASSERT_SAME_TYPE(decltype(s), std::string_view);
    assert(s == "ab");

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    std::basic_string_view w(L"abcdef", 4);
    ASSERT_SAME_TYPE(decltype(w), std::wstring_view);
    assert(w == L"abcd");
#endif
  }
  { // Testing (4)
    std::basic_string_view s("abc");
    ASSERT_SAME_TYPE(decltype(s), std::string_view);
    assert(s == "abc");

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    std::basic_string_view w(L"abcdef");
    ASSERT_SAME_TYPE(decltype(w), std::wstring_view);
    assert(w == L"abcdef");
#endif
  }

  return 0;
}
