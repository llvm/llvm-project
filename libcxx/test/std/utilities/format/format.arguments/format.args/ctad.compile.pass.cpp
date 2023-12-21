//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <format>

// template<class Context, class... Args>
//   basic_format_args(format-arg-store<Context, Args...>) -> basic_format_args<Context>;

#include <concepts>
#include <format>

#include "test_macros.h"

void test() {
  int i = 1;
  // Note the Standard way to create a format-arg-store is by using make_format_args.
  static_assert(std::same_as<decltype(std::basic_format_args(std::make_format_args(i))),
                             std::basic_format_args<std::format_context>>);

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  static_assert(std::same_as<decltype(std::basic_format_args(std::make_wformat_args(i))),
                             std::basic_format_args<std::wformat_context>>);

#endif
}
