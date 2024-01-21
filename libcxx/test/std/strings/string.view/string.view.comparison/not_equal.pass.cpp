//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: !stdlib=libc++ && (c++03 || c++11 || c++14)

// <string_view>

// template<class charT, class traits>
//   constexpr bool operator!=(basic_string_view<charT, traits> lhs, basic_string_view<charT, traits> rhs);
// (plus "sufficient additional overloads" to make implicit conversions work as intended)

#include <string_view>
#include <cassert>
#include <string>

#include "test_macros.h"
#include "constexpr_char_traits.h"
#include "make_string.h"

template <class T>
struct ConvertibleTo {
  T t_;
  TEST_CONSTEXPR explicit ConvertibleTo(T t) : t_(t) {}
  TEST_CONSTEXPR operator T() const { return t_; }
};

template <class SV>
TEST_CONSTEXPR_CXX14 bool test() {
  typedef typename SV::value_type CharT;
  typedef typename SV::traits_type Traits;

  // Test the behavior of the operator, both with and without implicit conversions.
  SV v[] = {
      SV(MAKE_CSTRING(CharT, "")),
      SV(MAKE_CSTRING(CharT, "abc")),
      SV(MAKE_CSTRING(CharT, "abcdef")),
      SV(MAKE_CSTRING(CharT, "acb")),
  };
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      // See http://eel.is/c++draft/string.view#tab:string.view.comparison.overloads
      bool expected = (i != j);
      assert((v[i] != v[j]) == expected);
      assert((v[i].data() != v[j]) == expected);
      assert((v[i] != v[j].data()) == expected);
      assert((ConvertibleTo<SV>(v[i]) != v[j]) == expected);
      assert((v[i] != ConvertibleTo<SV>(v[j])) == expected);

      if (TEST_STD_AT_LEAST_20_OR_RUNTIME_EVALUATED) {
        assert((std::basic_string<CharT, Traits>(v[i]) != v[j]) == expected);
        assert((v[i] != std::basic_string<CharT, Traits>(v[j])) == expected);
      }
    }
  }

  // Test its behavior with embedded null bytes.
  SV abc     = SV(MAKE_CSTRING(CharT, "abc"));
  SV abc0def = SV(MAKE_CSTRING(CharT, "abc\0def"), 7);
  SV abcdef  = SV(MAKE_CSTRING(CharT, "abcdef"));
  assert((abc != abc0def) == true);
  assert((abc != abcdef) == true);
  assert((abc0def != abc) == true);
  assert((abc0def != abcdef) == true);
  assert((abcdef != abc) == true);
  assert((abcdef != abc0def) == true);

  assert((abc.data() != abc0def) == true);
  assert((abc0def != abc.data()) == true);

  if (TEST_STD_AT_LEAST_20_OR_RUNTIME_EVALUATED) {
    assert((std::basic_string<CharT, Traits>(abc) != abc0def) == true);
    assert((abc0def != std::basic_string<CharT, Traits>(abc)) == true);
  }

  return true;
}

int main(int, char**) {
  test<std::string_view>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<std::wstring_view>();
#endif
#if TEST_STD_VER >= 11
  test<std::u16string_view>();
  test<std::u32string_view>();
#endif
#if TEST_STD_VER > 14
  static_assert(test<std::string_view>(), "");
#  ifndef TEST_HAS_NO_WIDE_CHARACTERS
  static_assert(test<std::wstring_view>(), "");
#  endif
  static_assert(test<std::u16string_view>(), "");
  static_assert(test<std::u32string_view>(), "");
#endif

#if TEST_STD_VER > 11
  test<std::basic_string_view<char, constexpr_char_traits<char>>>();
  static_assert(test<std::basic_string_view<char, constexpr_char_traits<char>>>(), "");
#endif

#if TEST_STD_VER > 17
  test<std::u8string_view>();
  static_assert(test<std::u8string_view>());
#endif

  return 0;
}
