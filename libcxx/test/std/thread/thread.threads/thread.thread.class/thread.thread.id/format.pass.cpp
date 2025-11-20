//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// UNSUPPORTED: no-threads

// UNSUPPORTED: GCC-ALWAYS_INLINE-FIXME

// <thread>

// template<class charT>
// struct formatter<thread::id, charT>;

// template<class FormatContext>
//   typename FormatContext::iterator
//     format(const T& r, FormatContext& ctx) const;

// Note this tests the basics of this function. It's tested in more detail in
// the format functions test.

#include <cassert>
#include <concepts>
#include <iterator>
#include <thread>

#include "test_format_context.h"
#include "test_macros.h"
#include "make_string.h"

#define SV(S) MAKE_STRING_VIEW(CharT, S)

template <class StringViewT>
void test_format(StringViewT expected, std::thread::id arg) {
  using CharT      = typename StringViewT::value_type;
  using String     = std::basic_string<CharT>;
  using OutIt      = std::back_insert_iterator<String>;
  using FormatCtxT = std::basic_format_context<OutIt, CharT>;

  const std::formatter<std::thread::id, CharT> formatter;

  String result;
  OutIt out             = std::back_inserter(result);
  FormatCtxT format_ctx = test_format_context_create<OutIt, CharT>(out, std::make_format_args<FormatCtxT>(arg));
  formatter.format(arg, format_ctx);
  assert(result == expected);
}

template <class CharT>
void test_fmt() {
#if !defined(__APPLE__) && !defined(__FreeBSD__)
  test_format(SV("0"), std::thread::id());
#else
  test_format(SV("0x0"), std::thread::id());
#endif
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
