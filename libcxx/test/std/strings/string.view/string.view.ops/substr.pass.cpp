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

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <string_view>

#include "test_macros.h"

template <typename CharT>
struct Test {
  typedef std::basic_string_view<CharT> (std::basic_string_view<CharT>::*Sub)(
      typename std::basic_string_view<CharT>::size_type, typename std::basic_string_view<CharT>::size_type) const;
};

template <typename CharT, typename Test<CharT>::Sub TestSub>
void testDetail(std::basic_string_view<CharT> sv, std::size_t n, size_t pos) {
  std::basic_string_view<CharT> sv1;
#ifdef TEST_HAS_NO_EXCEPTIONS
  if (pos > sv.size())
    return; // would throw if exceptions were enabled
  sv1 = (sv.*TestSub)(pos, n);
#else
  try {
    sv1 = (sv.*TestSub)(pos, n);
    assert(pos <= sv.size());
  } catch (const std::out_of_range&) {
    assert(pos > sv.size());
    return;
  }
#endif
  const std::size_t rlen = std::min(n, sv.size() - pos);
  assert(sv1.size() == rlen);
  for (std::size_t i = 0; i < rlen; ++i)
    assert(sv[pos + i] == sv1[i]);
}

template <typename CharT, typename Test<CharT>::Sub TestSub>
void testCases(const CharT* s) {
  std::basic_string_view<CharT> sv(s);

  testDetail<CharT, TestSub>(sv, 0, 0);
  testDetail<CharT, TestSub>(sv, 1, 0);
  testDetail<CharT, TestSub>(sv, 20, 0);
  testDetail<CharT, TestSub>(sv, sv.size(), 0);

  testDetail<CharT, TestSub>(sv, 100, 3);

  testDetail<CharT, TestSub>(sv, 0, std::basic_string_view<CharT>::npos);
  testDetail<CharT, TestSub>(sv, 2, std::basic_string_view<CharT>::npos);
  testDetail<CharT, TestSub>(sv, sv.size(), std::basic_string_view<CharT>::npos);

  testDetail<CharT, TestSub>(sv, sv.size() + 1, 0);
  testDetail<CharT, TestSub>(sv, sv.size() + 1, 1);
  testDetail<CharT, TestSub>(sv, sv.size() + 1, std::basic_string_view<CharT>::npos);
}

template <typename CharT>
void testSubs(const CharT* s) {
  testCases<CharT, &std::basic_string_view<CharT>::substr>(s);
#if TEST_STD_VER >= 26
  testCases<CharT, &std::basic_string_view<CharT>::subview>(s);
#endif
}

void test() {
  testSubs("ABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDE");
  testSubs("ABCDE");
  testSubs("a");
  testSubs("");

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  testSubs(
      L"ABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDE");
  testSubs(L"ABCDE");
  testSubs(L"a");
  testSubs(L"");
#endif

#if TEST_STD_VER >= 11
  testSubs(
      u"ABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDE");
  testSubs(u"ABCDE");
  testSubs(u"a");
  testSubs(u"");

  testSubs(
      U"ABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDE");
  testSubs(U"ABCDE");
  testSubs(U"a");
  testSubs(U"");
#endif
}

#if TEST_STD_VER >= 14
template <typename Test<char>::Sub TestSub>
constexpr void testConstexprDetail() {
  constexpr std::string_view sv{"ABCDE", 5};
  {
    constexpr std::string_view sv2 = (sv.*TestSub)(0, 3);

    static_assert(sv2.size() == 3, "");
    static_assert(sv2[0] == 'A', "");
    static_assert(sv2[1] == 'B', "");
    static_assert(sv2[2] == 'C', "");
  }

  {
    constexpr std::string_view sv2 = (sv.*TestSub)(3, 0);
    static_assert(sv2.size() == 0, "");
  }

  {
    constexpr std::string_view sv2 = (sv.*TestSub)(3, 3);
    static_assert(sv2.size() == 2, "");
    static_assert(sv2[0] == 'D', "");
    static_assert(sv2[1] == 'E', "");
  }
}

void test_constexpr() {
  testConstexprDetail<&std::string_view::substr>();
#  if TEST_STD_VER >= 26
  testConstexprDetail<&std::string_view::subview>();
#  endif
}
#endif

int main(int, char**) {
  test();
#if TEST_STD_VER >= 14
  test_constexpr();
#endif

  return 0;
}
