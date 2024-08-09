//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23
// UNSUPPORTED: GCC-ALWAYS_INLINE-FIXME
// UNSUPPORTED: clang-17
// XFAIL: apple-clang

// <format>

// template<class Visitor, class Context>
//   see below visit_format_arg(Visitor&& vis, basic_format_arg<Context> arg);

#include <format>
#include <tuple>

#include "test_macros.h"

template <typename CharT, class To, class From>
void test(From value) {
  using Context = std::basic_format_context<CharT*, CharT>;
  auto store    = std::make_format_args<Context>(value);
  std::basic_format_args<Context> format_args{store};

  // expected-warning-re@+1 1-2 {{std::basic_format_context{{.*}}' is deprecated}}
  std::ignore = std::visit_format_arg([]([[maybe_unused]] auto a) -> To { return {}; }, format_args.get(0));
}

void test() {
  test<char, bool>('a');
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t, bool>('a');
#endif
}
