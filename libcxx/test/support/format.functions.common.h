//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_SUPPORT_FORMAT_FUNCTIONS_COMMON_H
#define TEST_SUPPORT_FORMAT_FUNCTIONS_COMMON_H

// Contains the common part of the formatter tests for different papers.

#include <algorithm>
#include <charconv>
#include <format>

#include "make_string.h"

#define STR(S) MAKE_STRING(CharT, S)
#define SV(S) MAKE_STRING_VIEW(CharT, S)
#define CSTR(S) MAKE_CSTRING(CharT, S)

template <class T>
struct context {};

template <>
struct context<char> {
  using type = std::format_context;
};

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
template <>
struct context<wchar_t> {
  using type = std::wformat_context;
};
#endif

template <class T>
using context_t = typename context<T>::type;

// A user-defined type used to test the handle formatter.
enum class status : uint16_t { foo = 0xAAAA, bar = 0x5555, foobar = 0xAA55 };

// The formatter for a user-defined type used to test the handle formatter.
template <class CharT>
struct std::formatter<status, CharT> {
  int type = 0;

  constexpr auto parse(basic_format_parse_context<CharT>& parse_ctx) -> decltype(parse_ctx.begin()) {
    auto begin = parse_ctx.begin();
    auto end = parse_ctx.end();
    if (begin == end)
      return begin;

    switch (*begin) {
    case CharT('x'):
      break;
    case CharT('X'):
      type = 1;
      break;
    case CharT('s'):
      type = 2;
      break;
    case CharT('}'):
      return begin;
    default:
      throw_format_error("The format-spec type has a type not supported for a status argument");
    }

    ++begin;
    if (begin != end && *begin != CharT('}'))
      throw_format_error("The format-spec should consume the input or end with a '}'");

    return begin;
  }

  template <class Out>
  auto format(status s, basic_format_context<Out, CharT>& ctx) const -> decltype(ctx.out()) {
    const char* names[] = {"foo", "bar", "foobar"};
    char buffer[7];
    const char* begin = names[0];
    const char* end = names[0];
    switch (type) {
    case 0:
      begin = buffer;
      buffer[0] = '0';
      buffer[1] = 'x';
      end = std::to_chars(&buffer[2], std::end(buffer), static_cast<uint16_t>(s), 16).ptr;
      buffer[6] = '\0';
      break;

    case 1:
      begin = buffer;
      buffer[0] = '0';
      buffer[1] = 'X';
      end = std::to_chars(&buffer[2], std::end(buffer), static_cast<uint16_t>(s), 16).ptr;
      std::transform(static_cast<const char*>(&buffer[2]), end, &buffer[2], [](char c) {
        return static_cast<char>(std::toupper(c)); });
      buffer[6] = '\0';
      break;

    case 2:
      switch (s) {
      case status::foo:
        begin = names[0];
        break;
      case status::bar:
        begin = names[1];
        break;
      case status::foobar:
        begin = names[2];
        break;
      }
      end = begin + strlen(begin);
      break;
    }

    return std::copy(begin, end, ctx.out());
  }

private:
  void throw_format_error(const char* s) {
#ifndef TEST_HAS_NO_EXCEPTIONS
    throw std::format_error(s);
#else
    (void)s;
    std::abort();
#endif
  }
};

#endif // TEST_SUPPORT_FORMAT_FUNCTIONS_COMMON_H
