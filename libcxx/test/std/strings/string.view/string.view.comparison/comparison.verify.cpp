//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <string_view>

// template<class charT, class traits>
//   constexpr auto operator<=>(basic_string_view<charT, traits> lhs, basic_string_view<charT, traits> rhs);
//
// LWG 3432
// [string.view]/4
// Mandates: R denotes a comparison category type ([cmp.categories]).

#include <string_view>

#include "test_macros.h"

template <class CharT, class Ordering>
struct traits {
  typedef CharT char_type;
  typedef int int_type;
  typedef std::streamoff off_type;
  typedef std::streampos pos_type;
  typedef std::mbstate_t state_type;
  using comparison_category = Ordering;

  static constexpr void assign(char_type&, const char_type&) noexcept;
  static constexpr bool eq(char_type&, const char_type&) noexcept;
  static constexpr bool lt(char_type&, const char_type&) noexcept;

  static constexpr int compare(const char_type*, const char_type*, size_t) { return 0; }
  static constexpr size_t length(const char_type*);
  static constexpr const char_type* find(const char_type*, size_t, const char_type&);
  static constexpr char_type* move(char_type*, const char_type*, size_t);
  static constexpr char_type* copy(char_type*, const char_type*, size_t);
  static constexpr char_type* assign(char_type*, size_t, char_type);

  static constexpr int_type not_eof(int_type) noexcept;

  static constexpr char_type to_char_type(int_type) noexcept;

  static constexpr int_type to_int_type(char_type) noexcept;

  static constexpr bool eq_int_type(int_type, int_type) noexcept;

  static constexpr int_type eof() noexcept;
};

template <class CharT, class Ordering, bool Valid>
void test() {
  using type = std::basic_string_view<CharT, traits<CharT, Ordering>>;
  if constexpr (Valid)
    type{} <=> type{};
  else
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  // These diagnostics are issued for
  // - Every invalid ordering
  // - Every type
  // expected-error-re@string_view:* 15 {{{{(static_assert|static assertion)}} failed{{.*}}return type is not a comparison category type}}

  // This diagnostic is not issued for Ordering == void.
  // expected-error@string_view:* 10 {{no matching conversion for static_cast from}}
#else
  // One less test run when wchar_t is unavailable.
  // expected-error-re@string_view:* 12 {{{{(static_assert|static assertion)}} failed{{.*}}return type is not a comparison category type}}
  // expected-error@string_view:* 8 {{no matching conversion for static_cast from}}
#endif
    type{} <=> type{};
}

template <class CharT>
void test_all_orders() {
  test<CharT, std::strong_ordering, true>();
  test<CharT, std::weak_ordering, true>();
  test<CharT, std::partial_ordering, true>();

  test<CharT, void, false>();
  test<CharT, int, false>();
  test<CharT, float, false>();
}

void test_all_types() {
  test_all_orders<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test_all_orders<wchar_t>();
#endif
  test_all_orders<char8_t>();
  test_all_orders<char16_t>();
  test_all_orders<char32_t>();
}
