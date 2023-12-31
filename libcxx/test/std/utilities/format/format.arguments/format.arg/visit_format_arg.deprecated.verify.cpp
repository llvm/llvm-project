//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23
// UNSUPPORTED: GCC-ALWAYS_INLINE-FIXME

// <format>

// template<class Visitor, class Context>
//   see below visit_format_arg(Visitor&& vis, basic_format_arg<Context> arg);

#include <format>

void test() {
  // expected-warning@+1 {{std::basic_format_context<char *, char>>' is deprecated}}
  std::visit_format_arg([]([[maybe_unused]] auto a) -> char { return {}; },
                        std::basic_format_arg<std::basic_format_context<char*, char>>{});
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  // expected-warning@+1 {{std::basic_format_context<wchar_t *, wchar_t>>' is deprecated}}
  std::visit_format_arg([]([[maybe_unused]] auto a) -> wchar_t { return {}; },
                        std::basic_format_arg<std::basic_format_context<wchar_t*, wchar_t>>{});
#endif
}
