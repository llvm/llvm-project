//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: !stdlib=libc++ && (c++03 || c++11 || c++14)

// <string_view>

// constexpr basic_string_view substr(size_type pos = 0, size_type n = npos) const;
// constexpr basic_string_view subview(size_type pos = 0,
//                                     size_type n = npos) const;      // freestanding-deleted

// subview is alternative name of substr

// Throws: out_of_range if pos > size().
// Effects: Determines the effective length rlen of the string to reference as the smaller of n and size() - pos.
// Returns: basic_string_view(data()+pos, rlen).

#if 0
#  include <algorithm>
#  include <cassert>
#  include <string_view>

#  include "make_string.h"
#  include "test_macros.h"

#  define CS(S) MAKE_CSTRING(CharT, S)

template <typename CharT>
struct Test {
  typedef std::basic_string_view<CharT> (std::basic_string_view<CharT>::*Sub)(
      typename std::basic_string_view<CharT>::size_type = 0, typename std::basic_string_view<CharT>::size_type = npos) const;
};


template <typename CharT, typename Test<CharT>::Sub TestSub>
TEST_CONSTEXPR_CXX14 void testDetail(std::basic_string_view<CharT> sv, std::size_t n, size_t pos) {
  std::basic_string_view<CharT> sv1;
#  ifdef TEST_HAS_NO_EXCEPTIONS
  if (pos > sv.size())
    return; // would throw if exceptions were enabled
  sv1 = (sv.*TestSub)(pos, n);
#  else
  if (!TEST_IS_CONSTANT_EVALUATED) {
    try {
      sv1 = (sv.*TestSub)(pos, n);
      assert(pos <= sv.size());
    } catch (const std::out_of_range&) {
      assert(pos > sv.size());
      return;
    }
  }
#  endif
  const std::size_t rlen = std::min(n, sv.size() - pos);
  assert(sv1.size() == rlen);
  for (std::size_t i = 0; i < rlen; ++i)
    assert(sv[pos + i] == sv1[i]);
}

template <typename CharT, typename Test<CharT>::Sub TestSub>
TEST_CONSTEXPR_CXX14 void testCases(const CharT* s) {
  std::basic_string_view<CharT> sv(s);

  testDetail<CharT, TestSub>(sv, 0, 0);
  testDetail<CharT, TestSub>(sv, 1, 0);
  testDetail<CharT, TestSub>(sv, 20, 0);
  testDetail<CharT, TestSub>(sv, sv.size(), 0);

  testDetail<CharT, TestSub>(sv, 100, 3);

  testDetail<CharT, TestSub>(sv, 0, std::basic_string_view<CharT>::npos);
  testDetail<CharT, TestSub>(sv, 2, std::basic_string_view<CharT>::npos);
  testDetail<CharT, TestSub>(sv, sv.size(), std::basic_string_view<CharT>::npos);

  // Test if exceptions are thrown correctly.
  testDetail<CharT, TestSub>(sv, sv.size() + 1, 0);
  testDetail<CharT, TestSub>(sv, sv.size() + 1, 1);
  testDetail<CharT, TestSub>(sv, sv.size() + 1, std::basic_string_view<CharT>::npos);
}

template <typename CharT>
TEST_CONSTEXPR_CXX14 void testSubs(const CharT* s) {
  testCases<CharT, &std::basic_string_view<CharT>::substr>(s);
#  if TEST_STD_VER >= 26
  testCases<CharT, &std::basic_string_view<CharT>::subview>(s);
#  endif
}

#  define CS(S) MAKE_CSTRING(CharT, S)

template <typename CharT>
TEST_CONSTEXPR_CXX14 void test() {
  testSubs(
      CS("ABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDE"));
  testSubs(CS("ABCDE"));
  testSubs(CS("a"));
  testSubs(CS(""));
}

TEST_CONSTEXPR_CXX14 bool test() {
  test<char>();
#  ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#  endif
#  if TEST_STD_VER >= 11
#    ifndef TEST_HAS_NO_CHAR8_T
  test<char8_t>();
#    endif
  test<char16_t>();
  test<char32_t>();
#  endif

  return true;
}

int main(int, char**) {
  test();
#  if TEST_STD_VER >= 14
  // static_assert(test());
#  endif

  return 0;
}
#endif

#include <cassert>
#include <string>
#include <utility>

#include "constexpr_char_traits.h"
#include "make_string.h"
#include "test_macros.h"

#define CS(S) MAKE_CSTRING(CharT, S)

template <typename CharT, typename TraitsT>
struct Test {
  typedef std::basic_string_view<CharT, TraitsT> (std::basic_string_view<CharT, TraitsT>::*Sub)(
      typename std::basic_string_view<CharT, TraitsT>::size_type,
      typename std::basic_string_view<CharT, TraitsT>::size_type) const;
};

template <typename CharT, typename TraitsT, typename Test<CharT, TraitsT>::Sub TestSub>
TEST_CONSTEXPR_CXX14 void test() {
  const std::basic_string_view<CharT, TraitsT> sv = CS("Hello cruel world!");

  // With a default position and a character length.
  assert((sv.*TestSub)(0, std::basic_string_view<CharT, TraitsT>::npos) == CS("Hello cruel world!"));

  // With a explict position and a character length.
  assert((sv.*TestSub)(6, 5) == CS("cruel"));

  // From the beginning of the string with a explicit character length.
  assert((sv.*TestSub)(0, 5) == CS("Hello"));

  // To the end of string with the default character length.
  assert((sv.*TestSub)(12, std::basic_string_view<CharT, TraitsT>::npos) == CS("world!"));

  // From the beginning to the end of the string with explicit values.
  assert((sv.*TestSub)(0, sv.size()) == CS("Hello cruel world!"));

  // Test if exceptions are thrown correctly.
#ifndef TEST_HAS_NO_EXCEPTIONS
  if (!TEST_IS_CONSTANT_EVALUATED) {
    { // With a position that is out of range.
      try {
        (sv.*TestSub)(sv.size() + 1, std::basic_string_view<CharT, TraitsT>::npos);
        assert(false && "Expected std::out_of_range exception");
      } catch (const std::out_of_range&) {
        // Expected exception
      }
    }

    { // With a position that is out of range and a 0 character length.
      try {
        (sv.*TestSub)(sv.size() + 1, 0);
        assert(false && "Expected std::out_of_range exception");
      } catch (const std::out_of_range&) {
        // Expected exception
      }
    }

    { // With a position that is out of range and a some character length.
      try {
        (sv.*TestSub)(sv.size() + 1, 1);
        assert(false && "Expected std::out_of_range exception");
      } catch (const std::out_of_range&) {
        // Expected exception
      }
    }
  }
#endif
}

template <typename CharT>
TEST_CONSTEXPR_CXX14 void test() {
  ASSERT_SAME_TYPE(std::basic_string_view<CharT>, decltype(std::declval<std::basic_string_view<CharT> >().substr()));
  test<CharT, std::char_traits<CharT>, &std::basic_string_view<CharT, std::char_traits<CharT> >::substr>();
  test<CharT, constexpr_char_traits<CharT>, &std::basic_string_view<CharT, constexpr_char_traits<CharT> >::substr>();
#if TEST_STD_VER >= 26
  ASSERT_SAME_TYPE(std::basic_string_view<CharT>, decltype(std::declval<std::basic_string_view<CharT> >().subview()));
  test<CharT, std::char_traits<CharT>, &std::basic_string_view<CharT, std::char_traits<CharT>>::subview>();
  test<CharT, constexpr_char_traits<CharT>, &std::basic_string_view<CharT, constexpr_char_traits<CharT> >::subview>();
#endif
}

TEST_CONSTEXPR_CXX14 bool test() {
  test<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif
#ifndef TEST_HAS_NO_CHAR8_T
  test<char8_t>();
#endif
  test<char16_t>();
  test<char32_t>();

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 14
  static_assert(test());
#endif

  return 0;
}