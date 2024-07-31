//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// XFAIL: no-wide-characters

// <format>

// template<class... Args>
//   format-arg-store<wformat_context, Args...>
//   make_wformat_args(Args&... args);

#include <cassert>
#include <format>

#include "test_basic_format_arg.h"
#include "test_macros.h"

template <class... Args>
concept can_make_wformat_args = requires(Args&&... args) { std::make_wformat_args(std::forward<Args>(args)...); };

static_assert(can_make_wformat_args<int&>);
static_assert(!can_make_wformat_args<int>);
static_assert(!can_make_wformat_args<int&&>);

int main(int, char**) {
  int i                       = 1;
  char c                      = 'c';
  nullptr_t p                 = nullptr;
  bool b                      = false;
  [[maybe_unused]] auto store = std::make_wformat_args(i, p, b, c);

  LIBCPP_STATIC_ASSERT(
      std::same_as<decltype(store), std::__format_arg_store<std::wformat_context, int, nullptr_t, bool, char>>);

  return 0;
}
