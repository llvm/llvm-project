//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20
// UNSUPPORTED: no-localization

// <chrono>
// parse: string and pointer formats, optional outputs, constraints, and ADL.

#include <cassert>
#include <chrono>
#include <concepts>
#include <locale>
#include <sstream>
#include <string>
#include <utility>

#include "make_string.h"
#include "test_macros.h"

#define STR(S) MAKE_STRING(CharT, S)

template <class... Args>
concept CanParse = requires(Args&&... args) { std::chrono::parse(std::forward<Args>(args)...); };

namespace custom {
template <int Arity>
struct value {
  int count            = 0;
  bool abbrev_was_null = false;
};

// parse must return the input stream even when the ADL function returns void.
template <class CharT, class Traits>
void from_stream(std::basic_istream<CharT, Traits>& is, const CharT* fmt, value<3>& out) {
  assert(fmt[0] == CharT('#'));
  is >> out.count;
}

template <class CharT, class Traits, class Alloc>
std::basic_istream<CharT, Traits>&
from_stream(std::basic_istream<CharT, Traits>& is,
            const CharT* fmt,
            value<4>& out,
            std::basic_string<CharT, Traits, Alloc>* abbrev) {
  assert(fmt[0] == CharT('#'));
  assert(abbrev != nullptr);
  *abbrev = STR("custom");
  return is >> out.count;
}

template <class CharT, class Traits, class Alloc>
std::basic_istream<CharT, Traits>&
from_stream(std::basic_istream<CharT, Traits>& is,
            const CharT* fmt,
            value<5>& out,
            std::basic_string<CharT, Traits, Alloc>* abbrev,
            std::chrono::minutes* offset) {
  assert(fmt[0] == CharT('#'));
  assert(offset != nullptr);
  out.abbrev_was_null = abbrev == nullptr;
  if (abbrev)
    *abbrev = STR("custom");
  *offset = std::chrono::minutes{90};
  return is >> out.count;
}

struct not_parsable {};
} // namespace custom

template <class CharT, class Format, int Arity>
void check_constraints() {
  using String  = std::basic_string<CharT>;
  using Value   = custom::value<Arity>;
  using Minutes = std::chrono::minutes;
  // Without optional outputs, parse calls a three-argument from_stream.
  static_assert(CanParse<const Format&, Value&> == (Arity == 3));
  // An abbreviation output requires a four-argument from_stream.
  static_assert(CanParse<const Format&, Value&, String&> == (Arity == 4));
  // An offset output requires five arguments, including a null abbreviation pointer.
  static_assert(CanParse<const Format&, Value&, Minutes&> == (Arity == 5));
  // Both optional outputs require a five-argument from_stream.
  static_assert(CanParse<const Format&, Value&, String&, Minutes&> == (Arity == 5));
}

// Test overload participation for each ADL signature and invalid result arguments.
template <class CharT, class Format>
void test_constraints() {
  check_constraints<CharT, Format, 3>();
  check_constraints<CharT, Format, 4>();
  check_constraints<CharT, Format, 5>();
  // Reject a type without a suitable from_stream.
  static_assert(!CanParse<const Format&, custom::not_parsable&>);
  // This custom parser requires a mutable result.
  static_assert(!CanParse<const Format&, const custom::value<3>&>);
  // The result must be an lvalue.
  static_assert(!CanParse<const Format&, custom::value<3>>);
}

// Test ADL dispatch and output forwarding for all four parse argument lists.
template <class CharT, class Format>
void test_adl() {
  using String              = std::basic_string<CharT>;
  const auto format_storage = STR("#");
  const Format fmt{format_storage.c_str()};
  // Call the void-returning ADL parser and return the original input stream.
  {
    std::basic_istringstream<CharT> is(STR("7"));
    custom::value<3> result{};
    auto& returned = is >> std::chrono::parse(fmt, result);
    static_assert(std::same_as<decltype(returned), std::basic_istream<CharT>&>);
    assert(&returned == &is);
    assert(!is.fail());
    assert(result.count == 7);
  }
  // Forward only the abbreviation output to the four-argument ADL parser.
  {
    std::basic_istringstream<CharT> is(STR("7"));
    custom::value<4> result{};
    String abbrev;
    auto& returned = is >> std::chrono::parse(fmt, result, abbrev);
    static_assert(std::same_as<decltype(returned), std::basic_istream<CharT>&>);
    assert(&returned == &is);
    assert(!is.fail());
    assert(result.count == 7);
    assert(abbrev == STR("custom"));
  }
  // Forward the offset output with a null abbreviation pointer.
  {
    std::basic_istringstream<CharT> is(STR("7"));
    custom::value<5> result{};
    std::chrono::minutes offset{};
    auto& returned = is >> std::chrono::parse(fmt, result, offset);
    static_assert(std::same_as<decltype(returned), std::basic_istream<CharT>&>);
    assert(&returned == &is);
    assert(!is.fail());
    assert(result.count == 7);
    assert(result.abbrev_was_null);
    assert(offset == std::chrono::minutes{90});
  }
  // Forward both optional outputs to the five-argument ADL parser.
  {
    std::basic_istringstream<CharT> is(STR("7"));
    custom::value<5> result{};
    String abbrev;
    std::chrono::minutes offset{};
    auto& returned = is >> std::chrono::parse(fmt, result, abbrev, offset);
    static_assert(std::same_as<decltype(returned), std::basic_istream<CharT>&>);
    assert(&returned == &is);
    assert(!is.fail());
    assert(result.count == 7);
    assert(!result.abbrev_was_null);
    assert(abbrev == STR("custom"));
    assert(offset == std::chrono::minutes{90});
  }
}

// Test delegation to chrono parsing, optional outputs, and failure propagation.
template <class CharT, class Format>
void test_chrono_overloads() {
  using namespace std::chrono;
  const sys_seconds date    = sys_days{2026y / July / 20};
  const auto format_storage = STR("%F %T %Z %z");
  const Format fmt{format_storage.c_str()};
  // Apply the parsed UTC offset without requesting the optional outputs.
  {
    std::basic_istringstream<CharT> is(STR("2026-07-20 00:00:00 UTC +0130!"));
    is.imbue(std::locale::classic());
    sys_seconds result{};
    static_assert(std::same_as<decltype(is >> parse(fmt, result)), std::basic_istream<CharT>&>);
    auto& returned = is >> parse(fmt, result);
    assert(&returned == &is);
    assert(!is.fail());
    assert(result == date - 90min);
    assert(is.peek() == CharT('!'));
  }
  // Store both optional outputs and leave the trailing character unread.
  {
    std::basic_istringstream<CharT> is(STR("2026-07-20 00:00:00 UTC +0130!"));
    is.imbue(std::locale::classic());
    sys_seconds result{};
    std::basic_string<CharT> abbrev;
    minutes offset{};
    is >> parse(fmt, result, abbrev, offset);
    assert(!is.fail());
    assert(result == date - 90min);
    assert(is.peek() == CharT('!'));
    assert(abbrev == STR("UTC"));
    assert(offset == 90min);
  }

  // A parsing failure is visible on the original stream.
  std::basic_istringstream<CharT> is(STR("invalid"));
  is.imbue(std::locale::classic());
  sys_seconds result{};
  is >> parse(fmt, result);
  assert(is.fail());
}

template <class CharT>
void test() {
  test_constraints<CharT, const CharT*>();
  test_constraints<CharT, std::basic_string<CharT>>();
  test_adl<CharT, const CharT*>();
  test_adl<CharT, std::basic_string<CharT>>();
  test_chrono_overloads<CharT, const CharT*>();
  test_chrono_overloads<CharT, std::basic_string<CharT>>();
}

int main(int, char**) {
  test<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif
  return 0;
}
