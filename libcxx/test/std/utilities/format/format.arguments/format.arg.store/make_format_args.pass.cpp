//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <format>

// template<class Context = format_context, class... Args>
// format-arg-store<Context, Args...> make_format_args(Args&... args);

#include <cassert>
#include <format>
#include <iterator>
#include <string>

#include "test_basic_format_arg.h"
#include "test_macros.h"

template <class... Args>
concept can_make_format_args = requires(Args&&... args) { std::make_format_args(std::forward<Args>(args)...); };

static_assert(can_make_format_args<int&>);
static_assert(!can_make_format_args<int>);
static_assert(!can_make_format_args<int&&>);

int main(int, char**) {
  int i                       = 1;
  char c                      = 'c';
  nullptr_t p                 = nullptr;
  bool b                      = false;
  [[maybe_unused]] auto store = std::make_format_args(i, p, b, c);

  LIBCPP_STATIC_ASSERT(
      std::same_as<decltype(store), std::__format_arg_store<std::format_context, int, nullptr_t, bool, char>>);

  return 0;
}
