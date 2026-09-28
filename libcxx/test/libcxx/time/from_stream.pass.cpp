//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
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
  struct TestCase {
    std::basic_string<CharT> input;
    unsigned width;
    std::uint64_t limit;
    std::uint64_t value;
    unsigned count;
    bool overflow;
    CharT next;
  };
  const TestCase cases[] = {
      // Read ordinary digits, stopping at the width or a non-digit.
      {STR("12345"), 2, 12, 12, 2, false, CharT('3')},
      {STR("0012X"), 5, 12, 12, 4, false, CharT('X')},
      // No digits are consumed when the width is zero or the input is not numeric.
      {STR("12"), 0, 12, 0, 0, false, CharT('1')},
      {STR("X"), 1, 12, 0, 0, false, CharT('X')},
      // Stop as soon as the next digit would exceed the limit.
      {STR("12345"), 5, 12, 12, 3, true, CharT('4')},
      {STR("13X"), 3, 12, 1, 2, true, CharT('X')},
  };
  for (const auto& c : cases) {
    std::basic_istringstream<CharT> stream(c.input);
    std::ios_base::iostate state = std::ios_base::goodbit;
    std::chrono::__from_stream_parser<CharT, std::char_traits<CharT>> parser{stream, state};
    auto result = parser.__read_digits(c.width, c.limit);
    assert(result.__value == c.value);
    assert(result.__digits_read == c.count);
    assert(result.__overflow == c.overflow);
    assert(state == std::ios_base::goodbit); // Reporting failure is the caller's responsibility.
    assert(stream.peek() == c.next);
  }
}

void test_read_signed() {
  struct TestCase {
    std::string input;
    int expected;
  };
  const TestCase cases[] = {
      // Ordinary values, with and without an explicit sign.
      {"123", 123},
      {"+123", 123},
      {"-123", -123},
      // The negative limit has no positive counterpart in int.
      {std::to_string(std::numeric_limits<int>::min()), std::numeric_limits<int>::min()},
  };
  for (const auto& c : cases) {
    std::istringstream stream{c.input};
    int result                   = -1;
    std::ios_base::iostate state = std::ios_base::goodbit;
    std::chrono::__from_stream_parser<char, std::char_traits<char>> parser{stream, state};
    parser.__read_signed(static_cast<unsigned>(c.input.size()), result);
    assert(!(state & (std::ios_base::failbit | std::ios_base::badbit)));
    assert(result == c.expected);
  }
}

void test_field_queries() {
  using Fields         = std::chrono::__fields_storage;
  using Parts          = std::chrono::__fields_set;
  constexpr auto check = [] {
    Fields fields;
    assert(!fields.__has(Parts::__day));
    assert(!fields.__has_any(Parts::__day | Parts::__month));
    fields.__set(Parts::__utc_offset);
    assert(fields.__has(Parts::__utc_offset));
    assert(!fields.__has_exactly(Parts::__day));
    fields.__set(Parts::__day);
    // __has_only and __has_exactly ignore the UTC offset field.
    assert(fields.__has_only(Parts::__day));
    assert(fields.__has_exactly(Parts::__day));
    assert(fields.__has(Parts::__day | Parts::__utc_offset));
    assert(fields.__has_any(Parts::__day | Parts::__month));
    assert(!fields.__has(Parts::__day | Parts::__month));
    assert(fields.__has_only(Parts::__day | Parts::__month));
    assert(fields.__has_exactly(Parts::__day, Parts::__month));
    fields.__set(Parts::__month);
    assert(!fields.__has_only(Parts::__day));
    assert(!fields.__has_exactly(Parts::__day));
    assert(fields.__has_exactly(Parts::__day, Parts::__month));
    assert(fields.__has_exactly(Parts::__day | Parts::__month));
    return true;
  };
  static_assert(check());
  assert(check());
}

template <class CharT, class T>
void check_repeated_field(const std::basic_string<CharT>& format,
                          const std::basic_string<CharT>& matching,
                          const std::basic_string<CharT>& conflicting,
                          T expected) {
  {
    std::basic_istringstream<CharT> stream(matching);
    stream.imbue(std::locale::classic());
    T result{};
    std::chrono::from_stream(stream, format.c_str(), result);
    assert(!stream.fail());
    assert(result == expected);
  }
  {
    std::basic_istringstream<CharT> stream(conflicting);
    stream.imbue(std::locale::classic());
    T result{};
    std::chrono::from_stream(stream, format.c_str(), result);
    assert(stream.fail());
  }
}

template <class CharT>
void test_repeated_fields() {
  using namespace std::chrono;
  // libc++ accepts equal repetitions and rejects conflicting values.
  check_repeated_field(STR("%d %d"), STR("01 01"), STR("01 02"), day{1});

  // Compound formats and locale formats share the same parsed fields.
  check_repeated_field(STR("%F %x"), STR("2026-07-20 07/20/26"), STR("2026-07-20 07/21/26"), 2026y / July / 20);

  // Compare fractional values, not their spelling; locale seconds have no fraction.
  check_repeated_field(STR("%S %S"), STR("1.25 1.250"), STR("1.25 1.5"), 1250ms);
  check_repeated_field(STR("%S %X"), STR("1.0 00:00:01"), STR("1.5 00:00:01"), 1000ms);

  // Zone fields are checked even without output pointers.
  check_repeated_field(STR("%d %z %Ez"), STR("01 +0100 +1:00"), STR("01 +0100 +2:00"), day{1});
  check_repeated_field(STR("%d %Z %Z"), STR("01 UTC UTC"), STR("01 UTC GMT"), day{1});
}

int main(int, char**) {
  test_read_signed();
  test_field_queries();
  test_read_digits<char>();
  test_repeated_fields<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test_read_digits<wchar_t>();
  test_repeated_fields<wchar_t>();
#endif
  return 0;
}
