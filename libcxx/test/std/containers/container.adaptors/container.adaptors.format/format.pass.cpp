//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// UNSUPPORTED: GCC-ALWAYS_INLINE-FIXME

// [container.adaptors.format]
// For each of queue, priority_queue, and stack, the library provides the
// following formatter specialization where adaptor-type is the name of the
// template:
//
// template<class charT, class T, formattable<charT> Container, class... U>
//   struct formatter<adaptor-type<T, Container, U...>, charT>

// template<class FormatContext>
//   typename FormatContext::iterator
//     format(maybe-const-adaptor& r, FormatContext& ctx) const;

// Note this tests the basics of this function. It's tested in more detail in
// the format functions test.

#include <array>
#include <cassert>
#include <concepts>
#include <format>
#include <queue>
#include <stack>

#include "test_format_context.h"
#include "test_macros.h"
#include "make_string.h"

#define SV(S) MAKE_STRING_VIEW(CharT, S)

template <class StringViewT, class Arg>
void test_format(StringViewT expected, Arg arg) {
  using CharT      = typename StringViewT::value_type;
  using String     = std::basic_string<CharT>;
  using OutIt      = std::back_insert_iterator<String>;
  using FormatCtxT = std::basic_format_context<OutIt, CharT>;

  const std::formatter<Arg, CharT> formatter;

  String result;
  OutIt out             = std::back_inserter(result);
  FormatCtxT format_ctx = test_format_context_create<OutIt, CharT>(out, std::make_format_args<FormatCtxT>(arg));
  formatter.format(arg, format_ctx);
  assert(result == expected);
}

template <class CharT>
void test_fmt() {
  std::array input{1, 42, 99, 0};
  test_format(SV("[1, 42, 99, 0]"), std::queue<int>{input.begin(), input.end()});
  test_format(SV("[99, 42, 1, 0]"), std::priority_queue<int>{input.begin(), input.end()});
  test_format(SV("[1, 42, 99, 0]"), std::stack<int>{input.begin(), input.end()});
}

void test() {
  test_fmt<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test_fmt<wchar_t>();
#endif
}

int main(int, char**) {
  test();

  return 0;
}
