//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <string_view>

//  template <class It, class End>
//  constexpr basic_string_view(It begin, End end)

#include <string_view>
#include <cassert>
#include <iterator>

#include "make_string.h"
#include "test_iterators.h"

template<class It, class Sentinel, class CharT>
constexpr void test_construction(std::basic_string_view<CharT> val) {
  auto sv = std::basic_string_view<CharT>(It(val.data()), Sentinel(It(val.data() + val.size())));
  assert(sv.data() == val.data());
  assert(sv.size() == val.size());
}

template<class CharT>
constexpr void test_with_char() {
  const auto val = MAKE_STRING_VIEW(CharT, "test");
  test_construction<CharT*, CharT*>(val);
  test_construction<CharT*, const CharT*>(val);
  test_construction<const CharT*, CharT*>(val);
  test_construction<const CharT*, sized_sentinel<const CharT*>>(val);
  test_construction<contiguous_iterator<const CharT*>, contiguous_iterator<const CharT*>>(val);
  test_construction<contiguous_iterator<const CharT*>, sized_sentinel<contiguous_iterator<const CharT*>>>(val);
}

constexpr bool test() {
  test_with_char<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test_with_char<wchar_t>();
#endif
  test_with_char<char8_t>();
  test_with_char<char16_t>();
  test_with_char<char32_t>();

  return true;
}

#ifndef TEST_HAS_NO_EXCEPTIONS
template<class CharT>
struct ThrowingSentinel {
  friend bool operator==(const CharT*, ThrowingSentinel) noexcept { return true; }
  friend std::iter_difference_t<const CharT*> operator-(const CharT*, ThrowingSentinel) noexcept { return {}; }
  friend std::iter_difference_t<const CharT*> operator-(ThrowingSentinel, const CharT*) { throw 42; }
};

template <class CharT>
void test_throwing() {
  auto val = MAKE_STRING_VIEW(CharT, "test");
  try {
    (void)std::basic_string_view<CharT>(val.data(), ThrowingSentinel<CharT>());
    assert(false);
  } catch (int i) {
    assert(i == 42);
  }
}

void test_throwing() {
  test_throwing<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test_throwing<wchar_t>();
#endif
  test_throwing<char8_t>();
  test_throwing<char16_t>();
  test_throwing<char32_t>();
}
#endif

static_assert( std::is_constructible_v<std::string_view, const char*, char*>);
static_assert( std::is_constructible_v<std::string_view, char*, const char*>);
static_assert(!std::is_constructible_v<std::string_view, char*, void*>);               // not a sentinel
static_assert(!std::is_constructible_v<std::string_view, signed char*, signed char*>); // wrong char type
static_assert(!std::is_constructible_v<std::string_view, random_access_iterator<char*>, random_access_iterator<char*>>); // not contiguous
static_assert( std::is_constructible_v<std::string_view, contiguous_iterator<char*>, contiguous_iterator<char*>>);

int main(int, char**) {
  test();
  static_assert(test());

#ifndef TEST_HAS_NO_EXCEPTIONS
  test_throwing();
#endif

  return 0;
}
