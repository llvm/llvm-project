//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-format

// <format>

// template <class _Tp, class _CharT = char>
// struct formatter;

// Tests the enabled and disabled requirements for std::formatter.

#include <array>
#include <bitset>
#include <bitset>
#include <chrono>
#include <complex>
#include <concepts>
#include <deque>
#include <format>
#include <forward_list>
#include <list>
#include <memory>
#include <map>
#include <optional>
#include <queue>
#include <set>
#include <stack>
#include <span>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <valarray>
#include <variant>

#include "test_macros.h"

#ifndef TEST_HAS_NO_FILESYSTEM_LIBRARY
#  include <filesystem>
#endif
#ifndef TEST_HAS_NO_LOCALIZATION
#  include <regex>
#endif
#ifndef TEST_HAS_NO_THREADS
#  include <thread>
#endif

// Validate default template argument.
static_assert(std::same_as<std::formatter<int>, std::formatter<int, char>>);

// Concept for an enabled formatter.
//
// Since it's not possible to extract the T and CharT types from the formatter
// they are specified and the proper formatter is always intended to be
// defaulted.
//
// [formatter.requirements]/2
// A type F meets the Formatter requirements if it meets the BasicFormatter
// requirements and the expressions shown in Table 71 are valid and have the
// indicated semantics.
template <class T, class CharT, class F = std::formatter<T, CharT>>
concept enabled =
    // The basic BasicFormatter requirements:
    std::default_initializable<F> && std::copyable<F> && std::destructible<F> && std::swappable<F> &&
    // The expressions shown in Table 71
    requires(F f, std::basic_format_parse_context<CharT> pc, T u, std::basic_format_context<CharT*, CharT> fc) {
  { f.parse(pc) } -> std::same_as<typename decltype(pc)::iterator>;
  { f.format(u, fc) } -> std::same_as<typename decltype(fc)::iterator>;
};

// Concept for a disabled formatter.
//
// This uses the same template arguments as enable. This isn't required since
// the concept doesn't need to inspect T and CharT. This makes it easier for
// future changes. For example P2286 formatting ranges intents to change
// std::formatter<std::vector<int>> from disabled to enabled. The current way
// makes it easy to define a macro like
// #if TEST_STD_VER > 23
//   TEST_ENABLED_AFTER_CXX23(T, CharT) enabled<T, CharT>
// #else
//   TEST_ENABLED_AFTER_CXX23(T, CharT) disabled<T, CharT>
// #endif
template <class T, class CharT, class F = std::formatter<T, CharT>>
// [formatter.requirements]/5
// If F is a disabled specialization of formatter, these values are false:
concept disabled = !std::is_default_constructible_v<F> && !std::is_copy_constructible_v<F> &&
                   !std::is_move_constructible_v<F> && !std::is_copy_assignable_v<F> && !std::is_move_assignable_v<F>;

template <class T, class CharT>
void assert_formatter_is_disabled() {
  static_assert(disabled<T, CharT>);
}

template <class T, class CharT>
void assert_formatter_is_enabled() {
  // Only formatters for CharT == char || CharT == wchar_t are enabled for the
  // standard formatters. When CharT is a different type the formatter should
  // be disabled.
  if constexpr (std::same_as<CharT, char>
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
                || std::same_as<CharT, wchar_t>
#endif
  )
    static_assert(enabled<T, CharT>);
  else
    assert_formatter_is_disabled<T, CharT>();
}

// Tests for P0645 Text Formatting
template <class CharT>
void test_P0645() {
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  // Tests the special formatter that converts a char to a wchar_t.
  assert_formatter_is_enabled<char, wchar_t>();
#endif
  assert_formatter_is_enabled<CharT, CharT>();

  assert_formatter_is_enabled<CharT*, CharT>();
  assert_formatter_is_enabled<const CharT*, CharT>();
  assert_formatter_is_enabled<const CharT[42], CharT>();
  assert_formatter_is_enabled<std::basic_string<CharT>, CharT>();
  assert_formatter_is_enabled<std::basic_string_view<CharT>, CharT>();

  assert_formatter_is_enabled<bool, CharT>();

  assert_formatter_is_enabled<signed char, CharT>();
  assert_formatter_is_enabled<signed short, CharT>();
  assert_formatter_is_enabled<signed int, CharT>();
  assert_formatter_is_enabled<signed long, CharT>();
  assert_formatter_is_enabled<signed long long, CharT>();
#ifndef TEST_HAS_NO_INT128
  assert_formatter_is_enabled<__int128_t, CharT>();
#endif

  assert_formatter_is_enabled<unsigned char, CharT>();
  assert_formatter_is_enabled<unsigned short, CharT>();
  assert_formatter_is_enabled<unsigned int, CharT>();
  assert_formatter_is_enabled<unsigned long, CharT>();
  assert_formatter_is_enabled<unsigned long long, CharT>();
#ifndef TEST_HAS_NO_INT128
  assert_formatter_is_enabled<__uint128_t, CharT>();
#endif

  assert_formatter_is_enabled<float, CharT>();
  assert_formatter_is_enabled<double, CharT>();
  assert_formatter_is_enabled<long double, CharT>();

  assert_formatter_is_enabled<std::nullptr_t, CharT>();
  assert_formatter_is_enabled<void*, CharT>();
  assert_formatter_is_enabled<const void*, CharT>();
}

// Tests for P1361 Integration of chrono with text formatting
//
// Some tests are commented out since these types haven't been implemented in
// chrono yet. After P1361 has been implemented these formatters should be all
// enabled.
template <class CharT>
void test_P1361() {
  assert_formatter_is_disabled<std::chrono::microseconds, CharT>();

  assert_formatter_is_disabled<std::chrono::sys_time<std::chrono::microseconds>, CharT>();
  //assert_formatter_is_enabled<std::chrono::utc_time<std::chrono::microseconds>, CharT>();
  //assert_formatter_is_enabled<std::chrono::tai_time<std::chrono::microseconds>, CharT>();
  //assert_formatter_is_enabled<std::chrono::gps_time<std::chrono::microseconds>, CharT>();
  assert_formatter_is_disabled<std::chrono::file_time<std::chrono::microseconds>, CharT>();
  assert_formatter_is_disabled<std::chrono::local_time<std::chrono::microseconds>, CharT>();

  assert_formatter_is_disabled<std::chrono::day, CharT>();
  assert_formatter_is_disabled<std::chrono::month, CharT>();
  assert_formatter_is_disabled<std::chrono::year, CharT>();

  assert_formatter_is_disabled<std::chrono::weekday, CharT>();
  assert_formatter_is_disabled<std::chrono::weekday_indexed, CharT>();
  assert_formatter_is_disabled<std::chrono::weekday_last, CharT>();

  assert_formatter_is_disabled<std::chrono::month_day, CharT>();
  assert_formatter_is_disabled<std::chrono::month_day_last, CharT>();
  assert_formatter_is_disabled<std::chrono::month_weekday, CharT>();
  assert_formatter_is_disabled<std::chrono::month_weekday_last, CharT>();

  assert_formatter_is_disabled<std::chrono::year_month, CharT>();
  assert_formatter_is_disabled<std::chrono::year_month_day, CharT>();
  assert_formatter_is_disabled<std::chrono::year_month_day_last, CharT>();
  assert_formatter_is_disabled<std::chrono::year_month_weekday, CharT>();
  assert_formatter_is_disabled<std::chrono::year_month_weekday_last, CharT>();

  assert_formatter_is_disabled<std::chrono::hh_mm_ss<std::chrono::microseconds>, CharT>();

  //assert_formatter_is_enabled<std::chrono::sys_info, CharT>();
  //assert_formatter_is_enabled<std::chrono::local_info, CharT>();

  //assert_formatter_is_enabled<std::chrono::zoned_time, CharT>();
}

// Tests for P1636 Formatters for library types
//
// The paper hasn't been voted in so currently all formatters are disabled.
// TODO validate whether the test is correct after the paper has been accepted.
template <class CharT>
void test_P1636() {
  assert_formatter_is_disabled<std::basic_streambuf<CharT>, CharT>();
  assert_formatter_is_disabled<std::bitset<42>, CharT>();
  assert_formatter_is_disabled<std::complex<double>, CharT>();
  assert_formatter_is_disabled<std::error_code, CharT>();
#ifndef TEST_HAS_NO_FILESYSTEM_LIBRARY
  assert_formatter_is_disabled<std::filesystem::path, CharT>();
#endif
  assert_formatter_is_disabled<std::shared_ptr<int>, CharT>();
#ifndef TEST_HAS_NO_LOCALIZATION
  assert_formatter_is_disabled<std::sub_match<CharT*>, CharT>();
#endif
#ifndef TEST_HAS_NO_THREADS
  assert_formatter_is_disabled<std::thread::id, CharT>();
#endif
  assert_formatter_is_disabled<std::unique_ptr<int>, CharT>();
}

// Tests for P2286 Formatting ranges
//
// The paper hasn't been voted in so currently all formatters are disabled.
// TODO validate whether the test is correct after the paper has been accepted.
template <class CharT>
void test_P2286() {
  assert_formatter_is_disabled<std::array<int, 42>, CharT>();
  assert_formatter_is_disabled<std::vector<int>, CharT>();
  assert_formatter_is_disabled<std::deque<int>, CharT>();
  assert_formatter_is_disabled<std::forward_list<int>, CharT>();
  assert_formatter_is_disabled<std::list<int>, CharT>();

  assert_formatter_is_disabled<std::set<int>, CharT>();
  assert_formatter_is_disabled<std::map<int, int>, CharT>();
  assert_formatter_is_disabled<std::multiset<int>, CharT>();
  assert_formatter_is_disabled<std::multimap<int, int>, CharT>();

  assert_formatter_is_disabled<std::unordered_set<int>, CharT>();
  assert_formatter_is_disabled<std::unordered_map<int, int>, CharT>();
  assert_formatter_is_disabled<std::unordered_multiset<int>, CharT>();
  assert_formatter_is_disabled<std::unordered_multimap<int, int>, CharT>();

  assert_formatter_is_disabled<std::stack<int>, CharT>();
  assert_formatter_is_disabled<std::queue<int>, CharT>();
  assert_formatter_is_disabled<std::priority_queue<int>, CharT>();

  assert_formatter_is_disabled<std::span<int>, CharT>();

  assert_formatter_is_disabled<std::valarray<int>, CharT>();

  assert_formatter_is_disabled<std::pair<int, int>, CharT>();
  assert_formatter_is_disabled<std::tuple<int>, CharT>();
}

class c {
  void f();
  void fc() const;
  static void sf();
};
enum e { a };
enum class ec { a };
template <class CharT>
void test_disabled() {
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  assert_formatter_is_disabled<const char*, wchar_t>();
#endif
  assert_formatter_is_disabled<const char*, char8_t>();
  assert_formatter_is_disabled<const char*, char16_t>();
  assert_formatter_is_disabled<const char*, char32_t>();

  assert_formatter_is_disabled<c, CharT>();
  assert_formatter_is_disabled<const c, CharT>();
  assert_formatter_is_disabled<volatile c, CharT>();
  assert_formatter_is_disabled<const volatile c, CharT>();

  assert_formatter_is_disabled<e, CharT>();
  assert_formatter_is_disabled<const e, CharT>();
  assert_formatter_is_disabled<volatile e, CharT>();
  assert_formatter_is_disabled<const volatile e, CharT>();

  assert_formatter_is_disabled<ec, CharT>();
  assert_formatter_is_disabled<const ec, CharT>();
  assert_formatter_is_disabled<volatile ec, CharT>();
  assert_formatter_is_disabled<const volatile ec, CharT>();

  assert_formatter_is_disabled<int*, CharT>();
  assert_formatter_is_disabled<const int*, CharT>();
  assert_formatter_is_disabled<volatile int*, CharT>();
  assert_formatter_is_disabled<const volatile int*, CharT>();

  assert_formatter_is_disabled<c*, CharT>();
  assert_formatter_is_disabled<const c*, CharT>();
  assert_formatter_is_disabled<volatile c*, CharT>();
  assert_formatter_is_disabled<const volatile c*, CharT>();

  assert_formatter_is_disabled<e*, CharT>();
  assert_formatter_is_disabled<const e*, CharT>();
  assert_formatter_is_disabled<volatile e*, CharT>();
  assert_formatter_is_disabled<const volatile e*, CharT>();

  assert_formatter_is_disabled<ec*, CharT>();
  assert_formatter_is_disabled<const ec*, CharT>();
  assert_formatter_is_disabled<volatile ec*, CharT>();
  assert_formatter_is_disabled<const volatile ec*, CharT>();

  assert_formatter_is_disabled<void (*)(), CharT>();
  assert_formatter_is_disabled<void (c::*)(), CharT>();
  assert_formatter_is_disabled<void (c::*)() const, CharT>();

  assert_formatter_is_disabled<std::optional<int>, CharT>();
  assert_formatter_is_disabled<std::variant<int>, CharT>();

  assert_formatter_is_disabled<std::shared_ptr<c>, CharT>();
  assert_formatter_is_disabled<std::unique_ptr<c>, CharT>();

  assert_formatter_is_disabled<std::array<c, 42>, CharT>();
  assert_formatter_is_disabled<std::vector<c>, CharT>();
  assert_formatter_is_disabled<std::deque<c>, CharT>();
  assert_formatter_is_disabled<std::forward_list<c>, CharT>();
  assert_formatter_is_disabled<std::list<c>, CharT>();

  assert_formatter_is_disabled<std::set<c>, CharT>();
  assert_formatter_is_disabled<std::map<c, int>, CharT>();
  assert_formatter_is_disabled<std::multiset<c>, CharT>();
  assert_formatter_is_disabled<std::multimap<c, int>, CharT>();

  assert_formatter_is_disabled<std::unordered_set<c>, CharT>();
  assert_formatter_is_disabled<std::unordered_map<c, int>, CharT>();
  assert_formatter_is_disabled<std::unordered_multiset<c>, CharT>();
  assert_formatter_is_disabled<std::unordered_multimap<c, int>, CharT>();

  assert_formatter_is_disabled<std::stack<c>, CharT>();
  assert_formatter_is_disabled<std::queue<c>, CharT>();
  assert_formatter_is_disabled<std::priority_queue<c>, CharT>();

  assert_formatter_is_disabled<std::span<c>, CharT>();

  assert_formatter_is_disabled<std::valarray<c>, CharT>();

  assert_formatter_is_disabled<std::pair<c, int>, CharT>();
  assert_formatter_is_disabled<std::tuple<c>, CharT>();

  assert_formatter_is_disabled<std::optional<c>, CharT>();
  assert_formatter_is_disabled<std::variant<c>, CharT>();
}

template <class CharT>
void test() {
  test_P0645<CharT>();
  test_P1361<CharT>();
  test_P1636<CharT>();
  test_P2286<CharT>();
  test_disabled<CharT>();
}

void test() {
  test<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif
  test<char8_t>();
  test<char16_t>();
  test<char32_t>();

  test<int>();
}
