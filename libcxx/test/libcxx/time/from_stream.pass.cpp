//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20
// UNSUPPORTED: no-localization

#include <cassert>
#include <chrono>
#include <cstdint>
#include <limits>
#include <locale>
#include <sstream>
#include <string>

#include "make_string.h"
#include "test_macros.h"

#define STR(S) MAKE_STRING(CharT, S)

template <class CharT>
void test_read_digits() {
  using Traits             = std::char_traits<CharT>;
  constexpr auto max_value = std::numeric_limits<std::uint64_t>::max();
  struct TestCase {
    std::basic_string<CharT> input;
    unsigned width;
    std::uint64_t limit;
    std::uint64_t value;
    unsigned count;
    bool overflow;
    typename Traits::int_type next;
    std::ios_base::iostate state = std::ios_base::goodbit;
  };
  const TestCase cases[] = {
      // No digits consumed.
      // A zero width consumes nothing.
      {STR("12"), 0, 12, 0, 0, false, CharT('1')},
      // An initial non-digit consumes nothing.
      {STR("X"), 1, 12, 0, 0, false, CharT('X')},
      // Empty input reports EOF without consuming any digits.
      {STR(""), 1, 12, 0, 0, false, Traits::eof(), std::ios_base::eofbit},

      // Successful reads.
      // Stop at the width, leaving '3' unread.
      {STR("12345"), 2, 12, 12, 2, false, CharT('3')},
      // Accept leading zeros and stop before the non-digit.
      {STR("0012X"), 5, 12, 12, 4, false, CharT('X')},
      // Reaching EOF before the width preserves the parsed value and reports EOF.
      {STR("12"), 3, 12, 12, 2, false, Traits::eof(), std::ios_base::eofbit},

      // Overflow.
      // After reaching the limit, consume '3' and report overflow; retain 12.
      {STR("12345"), 5, 12, 12, 3, true, CharT('4')},
      // Exceed the limit without reaching it first; consume '3' and retain 1.
      {STR("13X"), 3, 12, 1, 2, true, CharT('X')},

      // uint64_t boundaries.
      // Accept the largest uint64_t value without overflow.
      {STR("18446744073709551615X"), 21, max_value, max_value, 20, false, CharT('X')},
      // One above the largest uint64_t value overflows; retain the prefix before the final digit.
      {STR("18446744073709551616X"), 21, max_value, max_value / 10, 20, true, CharT('X')},
  };
  for (const auto& c : cases) {
    std::basic_istringstream<CharT> stream(c.input);
    std::ios_base::iostate state = std::ios_base::goodbit;
    std::chrono::__from_stream_parser<CharT, std::char_traits<CharT>> parser{stream, state};
    auto result = parser.__read_digits(c.width, c.limit);
    assert(result.__value == c.value);
    assert(result.__digits_read == c.count);
    assert(result.__overflow == c.overflow);
    assert(state == c.state);
    assert(stream.peek() == c.next);
  }
}

template <class CharT>
void test_read_signed() {
  using Traits            = std::char_traits<CharT>;
  constexpr int sentinel  = 42;
  constexpr int min_value = std::numeric_limits<int>::min();
  constexpr int max_value = std::numeric_limits<int>::max();
  const auto make_input   = [](std::int64_t value) {
    const auto input = std::to_string(value);
    return std::basic_string<CharT>(input.begin(), input.end());
  };
  struct TestCase {
    std::basic_string<CharT> input;
    int expected;
    bool expected_fail = false;
  };
  const TestCase cases[] = {
      // Ordinary values.
      // Accept digits without a sign.
      {STR("123"), 123},
      // Accept an explicit positive sign.
      {STR("+123"), 123},
      // Apply a negative sign.
      {STR("-123"), -123},

      // Integer boundaries.
      // Accept the largest positive int.
      {make_input(max_value), max_value},
      // The negative limit has no positive counterpart in int.
      {make_input(min_value), min_value},
      // Reject a value above INT_MAX and preserve the output.
      {make_input(static_cast<std::int64_t>(max_value) + 1), sentinel, true},
      // Reject a value below INT_MIN and preserve the output.
      {make_input(static_cast<std::int64_t>(min_value) - 1), sentinel, true},

      // Missing digits preserve the output.
      // A negative sign alone is not an integer.
      {STR("-"), sentinel, true},
  };
  for (const auto& c : cases) {
    std::basic_istringstream<CharT> stream{c.input};
    int result                   = sentinel;
    std::ios_base::iostate state = std::ios_base::goodbit;
    std::chrono::__from_stream_parser<CharT, Traits> parser{stream, state};
    parser.__read_signed(std::numeric_limits<unsigned>::max(), result);
    assert(bool(state & std::ios_base::failbit) == c.expected_fail);
    assert(!(state & std::ios_base::badbit));
    assert(result == c.expected);
    assert(stream.peek() == Traits::eof());
  }
}

template <class CharT>
void test_width_overflow() {
  std::basic_ostringstream<CharT> format;
  format.imbue(std::locale::classic());
  // Append a zero to UINT_MAX to make the format width overflow unsigned.
  format << CharT('%') << std::numeric_limits<unsigned>::max() << STR("0S");

  // libc++ stores the width in unsigned and rejects overflow before reading input.
  std::basic_istringstream<CharT> stream{STR("1.25!")};
  std::chrono::milliseconds result{};
  std::chrono::from_stream(stream, format.str().c_str(), result);
  assert(stream.fail());
  // Clear the failure state and verify that no input was consumed.
  stream.clear();
  assert(stream.peek() == CharT('1'));
}

template <class CharT, class T>
void check(const std::basic_string<CharT>& input, const std::basic_string<CharT>& format, T expected) {
  std::basic_istringstream<CharT> stream(input);
  stream.imbue(std::locale::classic());
  T result{};
  std::chrono::from_stream(stream, format.c_str(), result);
  assert(!stream.fail());
  assert(result == expected);
}

template <class T, class CharT>
void check_failure(const std::basic_string<CharT>& input, const std::basic_string<CharT>& format) {
  std::basic_istringstream<CharT> stream(input);
  stream.imbue(std::locale::classic());
  T result{};
  std::chrono::from_stream(stream, format.c_str(), result);
  assert(stream.fail());
}

template <class CharT>
void test_whitespace() {
  // libc++ does not implicitly skip whitespace before %e.
  check_failure<std::chrono::year_month_day>(STR("2026-07- 7"), STR("%Y-%m-%e"));
}

template <class CharT>
void test_repeated_fields() {
  using namespace std::chrono;
  // libc++ accepts equal repetitions and rejects conflicting values.
  check(STR("01 01"), STR("%d %d"), day{1});
  check_failure<day>(STR("01 02"), STR("%d %d"));

  // Compound formats and locale formats share the same parsed fields.
  check(STR("2026-07-20 07/20/26"), STR("%F %x"), 2026y / July / 20);
  check_failure<year_month_day>(STR("2026-07-20 07/21/26"), STR("%F %x"));

  // Reject conflicting calendar, ordinal, and ISO week date information.
  check_failure<year_month_day>(STR("2026-02-15 Mon"), STR("%F %a"));
  check_failure<year_month_day>(STR("2026 032 03"), STR("%Y %j %m"));
  check_failure<year_month_day>(STR("2020-W53-5 2020"), STR("%G-W%V-%u %Y"));

  // Different representations of the year or hour must agree.
  check_failure<year>(STR("2026 25"), STR("%Y %y"));
  check_failure<minutes>(STR("14 01"), STR("%H %I"));

  // Repeated seconds must agree, including their fractional parts.
  check(STR("1.25 1.25"), STR("%S %S"), 1250ms);
  check_failure<milliseconds>(STR("1.25 1.5"), STR("%S %S"));

  // Locale seconds have no fraction.
  check(STR("1.0 00:00:01"), STR("%S %X"), 1000ms);
  check_failure<milliseconds>(STR("1.5 00:00:01"), STR("%S %X"));

  // Zone fields are checked even without output pointers.
  check(STR("01 +0100 +1:00"), STR("%d %z %Ez"), day{1});
  check_failure<day>(STR("01 +0100 +2:00"), STR("%d %z %Ez"));
  check(STR("01 UTC UTC"), STR("%d %Z %Z"), day{1});
  check_failure<day>(STR("01 UTC GMT"), STR("%d %Z %Z"));
}

int main(int, char**) {
  test_read_digits<char>();
  test_read_signed<char>();
  test_width_overflow<char>();
  test_whitespace<char>();
  test_repeated_fields<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test_read_digits<wchar_t>();
  test_read_signed<wchar_t>();
  test_width_overflow<wchar_t>();
  test_whitespace<wchar_t>();
  test_repeated_fields<wchar_t>();
#endif
  return 0;
}
