//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// XFAIL: availability-fp_to_chars-missing

// <format>

// [format.formatter.spec]:
// Each header that declares the template `formatter` provides the following
// enabled specializations:
// For each `charT`, for each cv-unqualified arithmetic type `ArithmeticT`
// other than char, wchar_t, char8_t, char16_t, or char32_t, a specialization
//    template<> struct formatter<ArithmeticT, charT>
//
// This file tests with `ArithmeticT = floating-point`, for each valid `charT`.
// Where `floating-point` is one of:
// - float
// - double
// - long double

#include <format>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <charconv>
#include <concepts>
#include <iterator>
#include <string>
#include <type_traits>

#include "test_format_context.h"
#include "test_macros.h"
#include "make_string.h"

#define STR(S) MAKE_STRING(CharT, S)

template <class CharT, class ArithmeticT>
void test(std::basic_string_view<CharT> fmt, ArithmeticT arg, std::basic_string<CharT> expected, std::size_t offset) {
  auto parse_ctx = std::basic_format_parse_context<CharT>(fmt);
  std::formatter<ArithmeticT, CharT> formatter;
  static_assert(std::semiregular<decltype(formatter)>);

  auto it = formatter.parse(parse_ctx);
  assert(it == fmt.end() - offset);

  std::basic_string<CharT> result;
  auto out = std::back_inserter(result);
  using FormatCtxT = std::basic_format_context<decltype(out), CharT>;

  FormatCtxT format_ctx =
      test_format_context_create<decltype(out), CharT>(out, std::make_format_args<FormatCtxT>(arg));
  formatter.format(arg, format_ctx);

  if (expected.empty()) {
    std::array<char, 128> buffer;
    expected.append(buffer.data(), std::to_chars(buffer.data(), buffer.data() + buffer.size(), arg).ptr);
  }

  assert(result == expected);
}

template <class StringT, class ArithmeticT>
void test_termination_condition(StringT f, ArithmeticT arg, StringT expected = {}) {
  // The format-spec is valid if completely consumed or terminates at a '}'.
  // The valid inputs all end with a '}'. The test is executed twice:
  // - first with the terminating '}',
  // - second consuming the entire input.
  using CharT = typename StringT::value_type;
  std::basic_string_view<CharT> fmt{f};
  assert(fmt.back() == CharT('}') && "Pre-condition failure");

  test(fmt, arg, expected, 1);
  fmt.remove_suffix(1);
  test(fmt, arg, expected, 0);
}

template <class CharT, class ArithmeticT>
void test_hex_lower_case_precision(ArithmeticT value) {
  {
    std::array<char, 25'000> buffer;
    char* end_ptr = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value, std::chars_format::hex, 20'000).ptr;
    std::size_t size = end_ptr - buffer.data();
    auto end = buffer.begin() + size;
    test_termination_condition(STR(".20000a}"), value, std::basic_string<CharT>{buffer.begin(), end});

    std::size_t unused = buffer.end() - end;
    std::fill_n(end, unused, '#');
    test_termination_condition(STR("#<25000.20000a}"), value, std::basic_string<CharT>{buffer.begin(), buffer.end()});
    std::rotate(buffer.begin(), buffer.end() - (unused / 2), buffer.end());
    test_termination_condition(STR("#^25000.20000a}"), value, std::basic_string<CharT>{buffer.begin(), buffer.end()});
    std::rotate(buffer.begin(), buffer.end() - ((unused + 1) / 2), buffer.end());
    test_termination_condition(STR("#>25000.20000a}"), value, std::basic_string<CharT>{buffer.begin(), buffer.end()});
    std::fill_n(buffer.begin(), unused, '0');
    if (std::signbit(value)) {
      buffer[0] = '-';
      buffer[unused] = '0';
    }
    test_termination_condition(STR("025000.20000a}"), value, std::basic_string<CharT>{buffer.begin(), buffer.end()});
  }

#ifndef TEST_HAS_NO_LOCALIZATION
  {
    std::array<char, 25'000> buffer;
    char* end_ptr = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value, std::chars_format::hex, 20'000).ptr;
    std::size_t size = end_ptr - buffer.data();
    auto end = buffer.begin() + size;
    test_termination_condition(STR(".20000La}"), value, std::basic_string<CharT>{buffer.begin(), end});

    std::size_t unused = buffer.end() - end;
    std::fill_n(end, unused, '#');
    test_termination_condition(STR("#<25000.20000La}"), value, std::basic_string<CharT>{buffer.begin(), buffer.end()});
    std::rotate(buffer.begin(), buffer.end() - (unused / 2), buffer.end());
    test_termination_condition(STR("#^25000.20000La}"), value, std::basic_string<CharT>{buffer.begin(), buffer.end()});
    std::rotate(buffer.begin(), buffer.end() - ((unused + 1) / 2), buffer.end());
    test_termination_condition(STR("#>25000.20000La}"), value, std::basic_string<CharT>{buffer.begin(), buffer.end()});
    std::fill_n(buffer.begin(), unused, '0');
    if (std::signbit(value)) {
      buffer[0] = '-';
      buffer[unused] = '0';
    }
    test_termination_condition(STR("025000.20000La}"), value, std::basic_string<CharT>{buffer.begin(), buffer.end()});
  }
#endif
}

template <class CharT, class ArithmeticT>
void test_hex_upper_case_precision(ArithmeticT value) {
  {
    std::array<char, 25'000> buffer;
    char* end_ptr = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value, std::chars_format::hex, 20'000).ptr;
    std::size_t size = end_ptr - buffer.data();
    auto end = buffer.begin() + size;
    std::transform(buffer.begin(), end, buffer.begin(), [](char c) { return std::toupper(c); });
    test_termination_condition(STR(".20000A}"), value, std::basic_string<CharT>{buffer.begin(), end});

    std::size_t unused = buffer.end() - end;
    std::fill_n(end, unused, '#');
    test_termination_condition(STR("#<25000.20000A}"), value, std::basic_string<CharT>{buffer.begin(), buffer.end()});
    std::rotate(buffer.begin(), buffer.end() - (unused / 2), buffer.end());
    test_termination_condition(STR("#^25000.20000A}"), value, std::basic_string<CharT>{buffer.begin(), buffer.end()});
    std::rotate(buffer.begin(), buffer.end() - ((unused + 1) / 2), buffer.end());
    test_termination_condition(STR("#>25000.20000A}"), value, std::basic_string<CharT>{buffer.begin(), buffer.end()});
    std::fill_n(buffer.begin(), unused, '0');
    if (std::signbit(value)) {
      buffer[0] = '-';
      buffer[unused] = '0';
    }
    test_termination_condition(STR("025000.20000A}"), value, std::basic_string<CharT>{buffer.begin(), buffer.end()});
  }

#ifndef TEST_HAS_NO_LOCALIZATION
  {
    std::array<char, 25'000> buffer;
    char* end_ptr = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value, std::chars_format::hex, 20'000).ptr;
    std::size_t size = end_ptr - buffer.data();
    auto end = buffer.begin() + size;
    std::transform(buffer.begin(), end, buffer.begin(), [](char c) { return std::toupper(c); });
    test_termination_condition(STR(".20000LA}"), value, std::basic_string<CharT>{buffer.begin(), end});

    std::size_t unused = buffer.end() - end;
    std::fill_n(end, unused, '#');
    test_termination_condition(STR("#<25000.20000LA}"), value, std::basic_string<CharT>{buffer.begin(), buffer.end()});
    std::rotate(buffer.begin(), buffer.end() - (unused / 2), buffer.end());
    test_termination_condition(STR("#^25000.20000LA}"), value, std::basic_string<CharT>{buffer.begin(), buffer.end()});
    std::rotate(buffer.begin(), buffer.end() - ((unused + 1) / 2), buffer.end());
    test_termination_condition(STR("#>25000.20000LA}"), value, std::basic_string<CharT>{buffer.begin(), buffer.end()});
    std::fill_n(buffer.begin(), unused, '0');
    if (std::signbit(value)) {
      buffer[0] = '-';
      buffer[unused] = '0';
    }
    test_termination_condition(STR("025000.20000LA}"), value, std::basic_string<CharT>{buffer.begin(), buffer.end()});
  }
#endif
}

template <class CharT, class ArithmeticT>
void test_scientific_lower_case_precision(ArithmeticT value) {
  {
    std::array<char, 25'000> buffer;
    char* end_ptr = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value, std::chars_format::scientific, 20'000).ptr;
    std::size_t size = end_ptr - buffer.data();
    auto end = buffer.begin() + size;
    test_termination_condition(STR(".20000e}"), value, std::basic_string<CharT>{buffer.begin(), end});

    std::size_t unused = buffer.end() - end;
    std::fill_n(end, unused, '#');
    test_termination_condition(STR("#<25000.20000e}"), value, std::basic_string<CharT>{buffer.begin(), buffer.end()});
    std::rotate(buffer.begin(), buffer.end() - (unused / 2), buffer.end());
    test_termination_condition(STR("#^25000.20000e}"), value, std::basic_string<CharT>{buffer.begin(), buffer.end()});
    std::rotate(buffer.begin(), buffer.end() - ((unused + 1) / 2), buffer.end());
    test_termination_condition(STR("#>25000.20000e}"), value, std::basic_string<CharT>{buffer.begin(), buffer.end()});
    std::fill_n(buffer.begin(), unused, '0');
    if (std::signbit(value)) {
      buffer[0] = '-';
      buffer[unused] = '0';
    }
    test_termination_condition(STR("025000.20000e}"), value, std::basic_string<CharT>{buffer.begin(), buffer.end()});
  }

#ifndef TEST_HAS_NO_LOCALIZATION
  {
    std::array<char, 25'000> buffer;
    char* end_ptr = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value, std::chars_format::scientific, 20'000).ptr;
    std::size_t size = end_ptr - buffer.data();
    auto end = buffer.begin() + size;
    test_termination_condition(STR(".20000Le}"), value, std::basic_string<CharT>{buffer.begin(), end});

    std::size_t unused = buffer.end() - end;
    std::fill_n(end, unused, '#');
    test_termination_condition(STR("#<25000.20000Le}"), value, std::basic_string<CharT>{buffer.begin(), buffer.end()});
    std::rotate(buffer.begin(), buffer.end() - (unused / 2), buffer.end());
    test_termination_condition(STR("#^25000.20000Le}"), value, std::basic_string<CharT>{buffer.begin(), buffer.end()});
    std::rotate(buffer.begin(), buffer.end() - ((unused + 1) / 2), buffer.end());
    test_termination_condition(STR("#>25000.20000Le}"), value, std::basic_string<CharT>{buffer.begin(), buffer.end()});
    std::fill_n(buffer.begin(), unused, '0');
    if (std::signbit(value)) {
      buffer[0] = '-';
      buffer[unused] = '0';
    }
    test_termination_condition(STR("025000.20000Le}"), value, std::basic_string<CharT>{buffer.begin(), buffer.end()});
  }
#endif
}

template <class CharT, class ArithmeticT>
void test_scientific_upper_case_precision(ArithmeticT value) {
  {
    std::array<char, 25'000> buffer;
    char* end_ptr = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value, std::chars_format::scientific, 20'000).ptr;
    std::size_t size = end_ptr - buffer.data();
    auto end = buffer.begin() + size;
    std::transform(buffer.begin(), end, buffer.begin(), [](char c) { return std::toupper(c); });
    test_termination_condition(STR(".20000E}"), value, std::basic_string<CharT>{buffer.begin(), end});

    std::size_t unused = buffer.end() - end;
    std::fill_n(end, unused, '#');
    test_termination_condition(STR("#<25000.20000E}"), value, std::basic_string<CharT>{buffer.begin(), buffer.end()});
    std::rotate(buffer.begin(), buffer.end() - (unused / 2), buffer.end());
    test_termination_condition(STR("#^25000.20000E}"), value, std::basic_string<CharT>{buffer.begin(), buffer.end()});
    std::rotate(buffer.begin(), buffer.end() - ((unused + 1) / 2), buffer.end());
    test_termination_condition(STR("#>25000.20000E}"), value, std::basic_string<CharT>{buffer.begin(), buffer.end()});
    std::fill_n(buffer.begin(), unused, '0');
    if (std::signbit(value)) {
      buffer[0] = '-';
      buffer[unused] = '0';
    }
    test_termination_condition(STR("025000.20000E}"), value, std::basic_string<CharT>{buffer.begin(), buffer.end()});
  }

#ifndef TEST_HAS_NO_LOCALIZATION
  {
    std::array<char, 25'000> buffer;
    char* end_ptr = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value, std::chars_format::scientific, 20'000).ptr;
    std::size_t size = end_ptr - buffer.data();
    auto end = buffer.begin() + size;
    std::transform(buffer.begin(), end, buffer.begin(), [](char c) { return std::toupper(c); });
    test_termination_condition(STR(".20000LE}"), value, std::basic_string<CharT>{buffer.begin(), end});

    std::size_t unused = buffer.end() - end;
    std::fill_n(end, unused, '#');
    test_termination_condition(STR("#<25000.20000LE}"), value, std::basic_string<CharT>{buffer.begin(), buffer.end()});
    std::rotate(buffer.begin(), buffer.end() - (unused / 2), buffer.end());
    test_termination_condition(STR("#^25000.20000LE}"), value, std::basic_string<CharT>{buffer.begin(), buffer.end()});
    std::rotate(buffer.begin(), buffer.end() - ((unused + 1) / 2), buffer.end());
    test_termination_condition(STR("#>25000.20000LE}"), value, std::basic_string<CharT>{buffer.begin(), buffer.end()});
    std::fill_n(buffer.begin(), unused, '0');
    if (std::signbit(value)) {
      buffer[0] = '-';
      buffer[unused] = '0';
    }
    test_termination_condition(STR("025000.20000LE}"), value, std::basic_string<CharT>{buffer.begin(), buffer.end()});
  }
#endif
}

template <class CharT, class ArithmeticT>
void test_fixed_lower_case_precision(ArithmeticT value) {
  {
    std::array<char, 25'000> buffer;
    char* end_ptr = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value, std::chars_format::fixed, 20'000).ptr;
    std::size_t size = end_ptr - buffer.data();
    auto end = buffer.begin() + size;
    test_termination_condition(STR(".20000f}"), value, std::basic_string<CharT>{buffer.begin(), end});

    std::size_t unused = buffer.end() - end;
    std::fill_n(end, unused, '#');
    test_termination_condition(STR("#<25000.20000f}"), value, std::basic_string<CharT>{buffer.begin(), buffer.end()});
    std::rotate(buffer.begin(), buffer.end() - (unused / 2), buffer.end());
    test_termination_condition(STR("#^25000.20000f}"), value, std::basic_string<CharT>{buffer.begin(), buffer.end()});
    std::rotate(buffer.begin(), buffer.end() - ((unused + 1) / 2), buffer.end());
    test_termination_condition(STR("#>25000.20000f}"), value, std::basic_string<CharT>{buffer.begin(), buffer.end()});
    std::fill_n(buffer.begin(), unused, '0');
    if (std::signbit(value)) {
      buffer[0] = '-';
      buffer[unused] = '0';
    }
    test_termination_condition(STR("025000.20000f}"), value, std::basic_string<CharT>{buffer.begin(), buffer.end()});
  }

#ifndef TEST_HAS_NO_LOCALIZATION
  {
    std::array<char, 25'000> buffer;
    char* end_ptr = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value, std::chars_format::fixed, 20'000).ptr;
    std::size_t size = end_ptr - buffer.data();
    auto end = buffer.begin() + size;
    test_termination_condition(STR(".20000Lf}"), value, std::basic_string<CharT>{buffer.begin(), end});

    std::size_t unused = buffer.end() - end;
    std::fill_n(end, unused, '#');
    test_termination_condition(STR("#<25000.20000Lf}"), value, std::basic_string<CharT>{buffer.begin(), buffer.end()});
    std::rotate(buffer.begin(), buffer.end() - (unused / 2), buffer.end());
    test_termination_condition(STR("#^25000.20000Lf}"), value, std::basic_string<CharT>{buffer.begin(), buffer.end()});
    std::rotate(buffer.begin(), buffer.end() - ((unused + 1) / 2), buffer.end());
    test_termination_condition(STR("#>25000.20000Lf}"), value, std::basic_string<CharT>{buffer.begin(), buffer.end()});
    std::fill_n(buffer.begin(), unused, '0');
    if (std::signbit(value)) {
      buffer[0] = '-';
      buffer[unused] = '0';
    }
    test_termination_condition(STR("025000.20000Lf}"), value, std::basic_string<CharT>{buffer.begin(), buffer.end()});
  }
#endif
}

template <class CharT, class ArithmeticT>
void test_fixed_upper_case_precision(ArithmeticT value) {
  {
    std::array<char, 25'000> buffer;
    char* end_ptr = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value, std::chars_format::fixed, 20'000).ptr;
    std::size_t size = end_ptr - buffer.data();
    auto end = buffer.begin() + size;
    std::transform(buffer.begin(), end, buffer.begin(), [](char c) { return std::toupper(c); });
    test_termination_condition(STR(".20000F}"), value, std::basic_string<CharT>{buffer.begin(), end});

    std::size_t unused = buffer.end() - end;
    std::fill_n(end, unused, '#');
    test_termination_condition(STR("#<25000.20000F}"), value, std::basic_string<CharT>{buffer.begin(), buffer.end()});
    std::rotate(buffer.begin(), buffer.end() - (unused / 2), buffer.end());
    test_termination_condition(STR("#^25000.20000F}"), value, std::basic_string<CharT>{buffer.begin(), buffer.end()});
    std::rotate(buffer.begin(), buffer.end() - ((unused + 1) / 2), buffer.end());
    test_termination_condition(STR("#>25000.20000F}"), value, std::basic_string<CharT>{buffer.begin(), buffer.end()});
    std::fill_n(buffer.begin(), unused, '0');
    if (std::signbit(value)) {
      buffer[0] = '-';
      buffer[unused] = '0';
    }
    test_termination_condition(STR("025000.20000F}"), value, std::basic_string<CharT>{buffer.begin(), buffer.end()});
  }

#ifndef TEST_HAS_NO_LOCALIZATION
  {
    std::array<char, 25'000> buffer;
    char* end_ptr = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value, std::chars_format::fixed, 20'000).ptr;
    std::size_t size = end_ptr - buffer.data();
    auto end = buffer.begin() + size;
    std::transform(buffer.begin(), end, buffer.begin(), [](char c) { return std::toupper(c); });
    test_termination_condition(STR(".20000LF}"), value, std::basic_string<CharT>{buffer.begin(), end});

    std::size_t unused = buffer.end() - end;
    std::fill_n(end, unused, '#');
    test_termination_condition(STR("#<25000.20000LF}"), value, std::basic_string<CharT>{buffer.begin(), buffer.end()});
    std::rotate(buffer.begin(), buffer.end() - (unused / 2), buffer.end());
    test_termination_condition(STR("#^25000.20000LF}"), value, std::basic_string<CharT>{buffer.begin(), buffer.end()});
    std::rotate(buffer.begin(), buffer.end() - ((unused + 1) / 2), buffer.end());
    test_termination_condition(STR("#>25000.20000LF}"), value, std::basic_string<CharT>{buffer.begin(), buffer.end()});
    std::fill_n(buffer.begin(), unused, '0');
    if (std::signbit(value)) {
      buffer[0] = '-';
      buffer[unused] = '0';
    }
    test_termination_condition(STR("025000.20000LF}"), value, std::basic_string<CharT>{buffer.begin(), buffer.end()});
  }
#endif
}

template <class CharT, class ArithmeticT>
void test_general_lower_case_precision(ArithmeticT value) {
  {
    std::array<char, 25'000> buffer;
    char* end_ptr = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value, std::chars_format::general, 20'000).ptr;
    std::size_t size = end_ptr - buffer.data();
    auto end = buffer.begin() + size;
    test_termination_condition(STR(".20000g}"), value, std::basic_string<CharT>{buffer.begin(), end});

    std::size_t unused = buffer.end() - end;
    std::fill_n(end, unused, '#');
    test_termination_condition(STR("#<25000.20000g}"), value, std::basic_string<CharT>{buffer.begin(), buffer.end()});
    std::rotate(buffer.begin(), buffer.end() - (unused / 2), buffer.end());
    test_termination_condition(STR("#^25000.20000g}"), value, std::basic_string<CharT>{buffer.begin(), buffer.end()});
    std::rotate(buffer.begin(), buffer.end() - ((unused + 1) / 2), buffer.end());
    test_termination_condition(STR("#>25000.20000g}"), value, std::basic_string<CharT>{buffer.begin(), buffer.end()});
    std::fill_n(buffer.begin(), unused, '0');
    if (std::signbit(value)) {
      buffer[0] = '-';
      buffer[unused] = '0';
    }
    test_termination_condition(STR("025000.20000g}"), value, std::basic_string<CharT>{buffer.begin(), buffer.end()});
  }

#ifndef TEST_HAS_NO_LOCALIZATION
  {
    std::array<char, 25'000> buffer;
    char* end_ptr = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value, std::chars_format::general, 20'000).ptr;
    std::size_t size = end_ptr - buffer.data();
    auto end = buffer.begin() + size;
    test_termination_condition(STR(".20000Lg}"), value, std::basic_string<CharT>{buffer.begin(), end});

    std::size_t unused = buffer.end() - end;
    std::fill_n(end, unused, '#');
    test_termination_condition(STR("#<25000.20000Lg}"), value, std::basic_string<CharT>{buffer.begin(), buffer.end()});
    std::rotate(buffer.begin(), buffer.end() - (unused / 2), buffer.end());
    test_termination_condition(STR("#^25000.20000Lg}"), value, std::basic_string<CharT>{buffer.begin(), buffer.end()});
    std::rotate(buffer.begin(), buffer.end() - ((unused + 1) / 2), buffer.end());
    test_termination_condition(STR("#>25000.20000Lg}"), value, std::basic_string<CharT>{buffer.begin(), buffer.end()});
    std::fill_n(buffer.begin(), unused, '0');
    if (std::signbit(value)) {
      buffer[0] = '-';
      buffer[unused] = '0';
    }
    test_termination_condition(STR("025000.20000Lg}"), value, std::basic_string<CharT>{buffer.begin(), buffer.end()});
  }
#endif
}

template <class CharT, class ArithmeticT>
void test_general_upper_case_precision(ArithmeticT value) {
  {
    std::array<char, 25'000> buffer;
    char* end_ptr = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value, std::chars_format::general, 20'000).ptr;
    std::size_t size = end_ptr - buffer.data();
    auto end = buffer.begin() + size;
    std::transform(buffer.begin(), end, buffer.begin(), [](char c) { return std::toupper(c); });
    test_termination_condition(STR(".20000G}"), value, std::basic_string<CharT>{buffer.begin(), end});

    std::size_t unused = buffer.end() - end;
    std::fill_n(end, unused, '#');
    test_termination_condition(STR("#<25000.20000G}"), value, std::basic_string<CharT>{buffer.begin(), buffer.end()});
    std::rotate(buffer.begin(), buffer.end() - (unused / 2), buffer.end());
    test_termination_condition(STR("#^25000.20000G}"), value, std::basic_string<CharT>{buffer.begin(), buffer.end()});
    std::rotate(buffer.begin(), buffer.end() - ((unused + 1) / 2), buffer.end());
    test_termination_condition(STR("#>25000.20000G}"), value, std::basic_string<CharT>{buffer.begin(), buffer.end()});
    std::fill_n(buffer.begin(), unused, '0');
    if (std::signbit(value)) {
      buffer[0] = '-';
      buffer[unused] = '0';
    }
    test_termination_condition(STR("025000.20000G}"), value, std::basic_string<CharT>{buffer.begin(), buffer.end()});
  }

#ifndef TEST_HAS_NO_LOCALIZATION
  {
    std::array<char, 25'000> buffer;
    char* end_ptr = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value, std::chars_format::general, 20'000).ptr;
    std::size_t size = end_ptr - buffer.data();
    auto end = buffer.begin() + size;
    std::transform(buffer.begin(), end, buffer.begin(), [](char c) { return std::toupper(c); });
    test_termination_condition(STR(".20000LG}"), value, std::basic_string<CharT>{buffer.begin(), end});

    std::size_t unused = buffer.end() - end;
    std::fill_n(end, unused, '#');
    test_termination_condition(STR("#<25000.20000LG}"), value, std::basic_string<CharT>{buffer.begin(), buffer.end()});
    std::rotate(buffer.begin(), buffer.end() - (unused / 2), buffer.end());
    test_termination_condition(STR("#^25000.20000LG}"), value, std::basic_string<CharT>{buffer.begin(), buffer.end()});
    std::rotate(buffer.begin(), buffer.end() - ((unused + 1) / 2), buffer.end());
    test_termination_condition(STR("#>25000.20000LG}"), value, std::basic_string<CharT>{buffer.begin(), buffer.end()});
    std::fill_n(buffer.begin(), unused, '0');
    if (std::signbit(value)) {
      buffer[0] = '-';
      buffer[unused] = '0';
    }
    test_termination_condition(STR("025000.20000LG}"), value, std::basic_string<CharT>{buffer.begin(), buffer.end()});
  }
#endif
}

template <class CharT, class ArithmeticT>
void test_value(ArithmeticT value) {
  test_hex_lower_case_precision<CharT>(value);
  test_hex_upper_case_precision<CharT>(value);

  test_scientific_lower_case_precision<CharT>(value);
  test_scientific_upper_case_precision<CharT>(value);

  test_fixed_lower_case_precision<CharT>(value);
  test_fixed_upper_case_precision<CharT>(value);

  test_general_lower_case_precision<CharT>(value);
  test_general_upper_case_precision<CharT>(value);
}

template <class ArithmeticT, class CharT>
void test_special_values() {
  using A = ArithmeticT;

  test_value<CharT>(-std::numeric_limits<A>::max());
  test_value<CharT>(A(-1.0));
  test_value<CharT>(-std::numeric_limits<A>::min());
  test_value<CharT>(-std::numeric_limits<A>::denorm_min());
  test_value<CharT>(A(-0.0));

  test_value<CharT>(A(0.0));
  test_value<CharT>(std::numeric_limits<A>::denorm_min());
  test_value<CharT>(A(1.0));
  test_value<CharT>(std::numeric_limits<A>::min());
  test_value<CharT>(std::numeric_limits<A>::max());
}

template <class ArithmeticT, class CharT>
void test_float_type() {
  using A = ArithmeticT;

  test_termination_condition(STR("}"), A(-std::numeric_limits<float>::max()));
  test_termination_condition(STR("}"), A(-std::numeric_limits<float>::min()));
  test_termination_condition(STR("}"), A(-0.0));

  test_termination_condition(STR("}"), A(0.0));
  test_termination_condition(STR("}"), A(std::numeric_limits<float>::min()));
  test_termination_condition(STR("}"), A(std::numeric_limits<float>::max()));
  if (sizeof(A) > sizeof(float)) {
    test_termination_condition(STR("}"), A(-std::numeric_limits<double>::max()));
    test_termination_condition(STR("}"), A(-std::numeric_limits<double>::min()));
    test_termination_condition(STR("}"), A(std::numeric_limits<double>::min()));
    test_termination_condition(STR("}"), A(std::numeric_limits<double>::max()));
  }
  if (sizeof(A) > sizeof(double)) {
    test_termination_condition(STR("}"), A(-std::numeric_limits<long double>::max()));
    test_termination_condition(STR("}"), A(-std::numeric_limits<long double>::min()));
    test_termination_condition(STR("}"), A(std::numeric_limits<long double>::min()));
    test_termination_condition(STR("}"), A(std::numeric_limits<long double>::max()));
  }

  // The results of inf and nan may differ from the result of to_chars.
  test_termination_condition(STR("}"), A(-std::numeric_limits<A>::infinity()), STR("-inf"));
  test_termination_condition(STR("}"), A(std::numeric_limits<A>::infinity()), STR("inf"));

  A nan = std::numeric_limits<A>::quiet_NaN();
  test_termination_condition(STR("}"), std::copysign(nan, -1.0), STR("-nan"));
  test_termination_condition(STR("}"), nan, STR("nan"));

  // TODO FMT Enable long double testing
  if constexpr (!std::same_as<A, long double>)
    test_special_values<A, CharT>();
}

template <class CharT>
void test_all_float_types() {
  test_float_type<float, CharT>();
  test_float_type<double, CharT>();
  test_float_type<long double, CharT>();
}

int main(int, char**) {
  test_all_float_types<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test_all_float_types<wchar_t>();
#endif

  return 0;
}
