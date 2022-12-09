//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template <class View, class Pattern>
// class std::ranges::lazy_split_view;
//
// These test check the output `lazy_split_view` produces for a variety of inputs, including many corner cases, with no
// restrictions on which member functions can be called.

#include <ranges>

#include <algorithm>
#include <array>
#include <cassert>
#include <concepts>
#include <map>
#include <string_view>
#include <string>
#include <utility>
#include <vector>
#include "types.h"

// Basic utility to convert a range to a string-like type. This handles ranges
// that do not contain character types and can work with non-contiguous inputs.
template <class Char>
class BasicSmallString {
  std::vector<Char> buffer_{};

public:
  constexpr BasicSmallString(std::basic_string_view<Char> v)
    requires (std::same_as<Char, char> ||
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
              std::same_as<Char, wchar_t> ||
#endif
              std::same_as<Char, char8_t> ||
              std::same_as<Char, char16_t> ||
              std::same_as<Char, char32_t>)
    : buffer_(v.begin(), v.end())
  {}

  template <class I, class S>
  constexpr BasicSmallString(I b, const S& e) {
    for (; b != e; ++b) {
      buffer_.push_back(*b);
    }
  }

  template <std::ranges::range R>
  constexpr BasicSmallString(R&& from) : BasicSmallString(from.begin(), from.end()) {}

  friend constexpr bool operator==(const BasicSmallString& lhs, const BasicSmallString& rhs) {
    return lhs.buffer_ == rhs.buffer_;
  }
};

template <std::ranges::view View, std::ranges::range Expected>
constexpr bool is_equal(View& view, const Expected& expected) {
  using Char = std::ranges::range_value_t<std::ranges::range_value_t<View>>;
  using Str = BasicSmallString<Char>;

  auto actual_it = view.begin();
  auto expected_it = expected.begin();
  for (; actual_it != view.end() && expected_it != expected.end(); ++actual_it, ++expected_it) {
    if (Str(*actual_it) != Str(*expected_it))
      return false;
  }

  return actual_it == view.end() && expected_it == expected.end();
}

template <class T, class Separator, class U, size_t M>
constexpr bool test_function_call(T&& input, Separator&& separator, std::array<U, M> expected) {
  std::ranges::lazy_split_view v(input, separator);
  return is_equal(v, expected);
}

template <class T, class Separator, class U, size_t M>
constexpr bool test_with_piping(T&& input, Separator&& separator, std::array<U, M> expected) {
  auto expected_it = expected.begin();
  for (auto e : input | std::ranges::views::lazy_split(separator)) {
    if (expected_it == expected.end())
      return false;
    if (!std::ranges::equal(e, *expected_it))
      return false;

    ++expected_it;
  }

  return expected_it == expected.end();
}

constexpr bool test_l_r_values() {
  using namespace std::string_view_literals;

  // Both lvalues and rvalues can be used as input.
  {
    // Lvalues.
    {
      auto input = "abc"sv;
      auto sep = " "sv;
      [[maybe_unused]] std::ranges::lazy_split_view v(input, sep);
    }

    // Const lvalues.
    {
      const auto input = "abc"sv;
      const auto sep = " "sv;
      [[maybe_unused]] std::ranges::lazy_split_view v(input, sep);
    }

    // Rvalues.
    {
      auto input = "abc"sv;
      auto sep = " "sv;
      [[maybe_unused]] std::ranges::lazy_split_view v(std::move(input), std::move(sep));
    }

    // Const rvalues.
    {
      const auto input = "abc"sv;
      const auto sep = " "sv;
      [[maybe_unused]] std::ranges::lazy_split_view v(std::move(input), std::move(sep));
    }
  }

  return true;
}

constexpr bool test_string_literal_separator() {
  using namespace std::string_view_literals;

  // Splitting works as expected when the separator is a single character literal.
  {
    std::ranges::lazy_split_view v("abc def"sv, ' ');
    assert(is_equal(v, std::array{"abc"sv, "def"sv}));
  }

  // Counterintuitively, a seemingly equivalent separator expressed as a string literal doesn't match anything. This is
  // because of the implicit terminating null in the literal.
  {
    std::ranges::lazy_split_view v("abc def"sv, " ");
    assert(is_equal(v, std::array{"abc def"sv}));
  }

  // To illustrate the previous point further, the separator is actually a two-character string literal: `{' ', '\0'}`.
  // Should the input string contain that two-character sequence, the separator would match.
  {
    std::ranges::lazy_split_view v("abc \0def"sv, " ");
    assert(is_equal(v, std::array{"abc"sv, "def"sv}));
  }

  return true;
}

// Make sure that a string literal and a `string_view` produce the same results (which isn't always the case, see
// below).
template <class T>
constexpr std::string_view sv(T&& str) {
  return std::string_view(str);
};

template <class T, class Separator, class U, size_t M>
constexpr void test_one(T&& input, Separator&& separator, std::array<U, M> expected) {
  assert(test_function_call(input, separator, expected));
  assert(test_with_piping(input, separator, expected));

  // In addition to the `(ForwardView, ForwardView)` case, test the `(ForwardView, tiny-range)` and `(InputView,
  // tiny-range)` cases (all of which have unique code paths).
  if constexpr (std::is_same_v<std::remove_reference_t<Separator>, char>) {
    assert(test_function_call(CopyableView(input), ForwardTinyView(separator), expected));
    assert(test_with_piping(CopyableView(input), ForwardTinyView(separator), expected));

    assert(test_function_call(InputView(input), ForwardTinyView(separator), expected));
    assert(test_with_piping(InputView(input), ForwardTinyView(separator), expected));
  }
}

constexpr bool test_string_literals() {
  // These tests show characteristic examples of how using string literals with `lazy_split_view` produces unexpected
  // results due to the implicit terminating null that is treated as part of the range.

  using namespace std::string_view_literals;

  char short_sep = ' ';
  auto long_sep = "12"sv;

  // When splitting a string literal, only the last segment will be null-terminated (getting the terminating null from
  // the original range).
  {
    std::array expected = {"abc"sv, std::string_view("def", sizeof("def"))};

    assert(test_function_call("abc def", short_sep, expected));
    assert(test_with_piping("abc def", short_sep, expected));
    assert(test_function_call("abc12def", long_sep, expected));
    assert(test_with_piping("abc12def", long_sep, expected));
  }

  // Empty string.
  {
    // Because an empty string literal contains an implicit terminating null, the output will contain one segment.
    std::array expected = {std::string_view("", 1)};

    assert(test_function_call("", short_sep, expected));
    assert(test_with_piping("", short_sep, expected));
    assert(test_function_call("", long_sep, expected));
    assert(test_with_piping("", long_sep, expected));
  }

  // Terminating null in the separator -- the character literal `' '` and the seemingly equivalent string literal `" "`
  // are treated differently due to the presence of an implicit `\0` in the latter.
  {
    const char input[] = "abc def";
    std::array expected_unsplit = {std::string_view(input, sizeof(input))};
    std::array expected_split = {"abc"sv, std::string_view("def", sizeof("def"))};

    assert(test_function_call(input, " ", expected_unsplit));
    assert(test_function_call("abc \0def", " ", expected_split));
    // Note: string literals don't work with piping because arrays decay to pointers, and pointers don't model `range`.
  }

  // Empty separator.
  {
    auto empty_sep = ""sv;
    std::array expected = {"a"sv, "b"sv, "c"sv, "\0"sv};

    assert(test_function_call("abc", empty_sep, expected));
    assert(test_with_piping("abc", empty_sep, expected));
  }

  return true;
}

bool test_nontrivial_characters() {
  // Try a deliberately heavyweight "character" type to see if it triggers any corner cases.

  using Map = std::map<std::string, int>;
  using Vec = std::vector<Map>;

  Map sep = {{"yyy", 999}};
  Map m1 = {
    {"a", 1},
    {"bc", 2},
  };
  Map m2 = {
    {"def", 3},
  };
  Map m3 = {
    {"g", 4},
    {"hijk", 5},
  };

  Vec expected1 = {m1, m2};
  Vec expected2 = {m3};

  std::ranges::lazy_split_view v(Vec{m1, m2, sep, m3}, sep);

  // Segment 1: {m1, m2}
  auto outer = v.begin();
  assert(outer != v.end());
  auto inner = (*outer).begin();
  assert(*inner++ == m1);
  assert(*inner++ == m2);
  assert(inner == (*outer).end());

  // Segment 2: {m3}
  ++outer;
  assert(outer != v.end());
  inner = (*outer).begin();
  assert(*inner++ == m3);
  assert(inner == (*outer).end());

  ++outer;
  assert(outer == v.end());

  return true;
}

constexpr bool main_test() {
  using namespace std::string_view_literals;

  char short_sep = ' ';
  auto long_sep = "12"sv;

  // One separator.
  {
    std::array expected = {"abc"sv, "def"sv};
    test_one("abc def"sv, short_sep, expected);
    test_one("abc12def"sv, long_sep, expected);
  }

  // Several separators in a row.
  {
    std::array expected = {"abc"sv, ""sv, ""sv, ""sv, "def"sv};
    test_one("abc    def"sv, short_sep, expected);
    test_one("abc12121212def"sv, long_sep, expected);
  }

  // Trailing separator.
  {
    std::array expected = {"abc"sv, "def"sv, ""sv};
    test_one("abc def "sv, short_sep, expected);
    test_one("abc12def12"sv, long_sep, expected);
  }

  // Leading separator.
  {
    std::array expected = {""sv, "abc"sv, "def"sv};
    test_one(" abc def"sv, short_sep, expected);
    test_one("12abc12def"sv, long_sep, expected);
  }

  // No separator.
  {
    std::array expected = {"abc"sv};
    test_one("abc"sv, short_sep, expected);
    test_one("abc"sv, long_sep, expected);
  }

  // Input consisting of a single separator.
  {
    std::array expected = {""sv, ""sv};
    test_one(" "sv, short_sep, expected);
    test_one("12"sv, long_sep, expected);
  }

  // Input consisting of only separators.
  {
    std::array expected = {""sv, ""sv, ""sv, ""sv};
    test_one("   "sv, short_sep, expected);
    test_one("121212"sv, long_sep, expected);
  }

  // The separator and the string use the same character only.
  {
    auto overlapping_sep = "aaa"sv;
    std::array expected = {""sv, "aa"sv};
    test_one("aaaaa"sv, overlapping_sep, expected);
  }

  // Many redundant separators.
  {
    std::array expected = {""sv, ""sv, "abc"sv, ""sv, ""sv, "def"sv, ""sv, ""sv};
    test_one("  abc   def  "sv, short_sep, expected);
    test_one("1212abc121212def1212"sv, long_sep, expected);
  }

  // Separators after every character.
  {
    std::array expected = {""sv, "a"sv, "b"sv, "c"sv, ""sv};
    test_one(" a b c "sv, short_sep, expected);
    test_one("12a12b12c12"sv, long_sep, expected);
  }

  // Overlap between the separator and the string (see https://wg21.link/lwg3505).
  {
    auto overlapping_sep = "ab"sv;
    std::array expected = {"a"sv, "aa"sv, ""sv, "b"sv};
    test_one("aabaaababb"sv, overlapping_sep, expected);
  }

  // Empty input.
  {
    std::array<std::string_view, 0> expected = {};
    test_one(""sv, short_sep, expected);
    test_one(""sv, long_sep, expected);
  }

  // Empty separator.
  {
    auto empty_sep = ""sv;
    std::array expected = {"a"sv, "b"sv, "c"sv};
    test_one("abc"sv, empty_sep, expected);
    test_one("abc"sv, empty_sep, expected);
  }

  // Terminating null as a separator.
  {
    std::array expected = {"abc"sv, "def"sv};
    test_one("abc\0def"sv, '\0', expected);
    test_one("abc\0\0def"sv, "\0\0"sv, expected);
  }

  // Different character types.
  {
    // `char`.
    test_function_call("abc def", ' ', std::array{"abc", "def"});
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    // `wchar_t`.
    test_function_call(L"abc def", L' ', std::array{L"abc", L"def"});
#endif
    // `char8_t`.
    test_function_call(u8"abc def", u8' ', std::array{u8"abc", u8"def"});
    // `char16_t`.
    test_function_call(u"abc def", u' ', std::array{u"abc", u"def"});
    // `char32_t`.
    test_function_call(U"abc def", U' ', std::array{U"abc", U"def"});
  }

  // Non-character input.
  {
    std::array expected = {std::array{1, 2, 3}, std::array{4, 5, 6}};
    test_one(std::array{1, 2, 3, 0, 4, 5, 6}, 0, expected);
    test_one(std::array{1, 2, 3, 0, 0, 0, 4, 5, 6}, std::array{0, 0, 0}, expected);
  }

  return true;
}

int main(int, char**) {
  main_test();
  static_assert(main_test());

  test_string_literals();
  static_assert(test_string_literals());

  test_l_r_values();
  static_assert(test_l_r_values());

  test_string_literal_separator();
  static_assert(test_string_literal_separator());

  // Note: map is not `constexpr`, so this test is runtime-only.
  test_nontrivial_characters();

  return 0;
}
