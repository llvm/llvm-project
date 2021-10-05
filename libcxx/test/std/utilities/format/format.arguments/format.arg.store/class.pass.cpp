//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: libcpp-has-no-incomplete-format

// <format>

// template<class Context, class... Args>
// struct format-arg-store {      // exposition only
//   array<basic_format_arg<Context>, sizeof...(Args)> args;
// };
//
// Note more testing is done in the unit test for:
// template<class Visitor, class Context>
//   see below visit_format_arg(Visitor&& vis, basic_format_arg<Context> arg);

#include <format>
#include <cassert>
#include <cstddef>
#include <type_traits>

#include "test_macros.h"

template <class CharT>
void test() {
  using Context = std::basic_format_context<CharT*, CharT>;
  {
    auto store = std::make_format_args<Context>();
    LIBCPP_STATIC_ASSERT(
        std::is_same_v<decltype(store), std::__format_arg_store<Context>>);
    LIBCPP_STATIC_ASSERT(
        std::is_same_v<decltype(store.__args),
                       std::array<std::basic_format_arg<Context>, 0>>);
    LIBCPP_ASSERT(store.__args.size() == 0);
  }
  {
    auto store = std::make_format_args<Context>(1);
    LIBCPP_STATIC_ASSERT(
        std::is_same_v<decltype(store), std::__format_arg_store<Context, int>>);
    LIBCPP_STATIC_ASSERT(
        std::is_same_v<decltype(store.__args),
                       std::array<std::basic_format_arg<Context>, 1>>);
    LIBCPP_ASSERT(store.__args.size() == 1);
  }
  {
    auto store = std::make_format_args<Context>(1, 'c');
    LIBCPP_STATIC_ASSERT(
        std::is_same_v<decltype(store),
                       std::__format_arg_store<Context, int, char>>);
    LIBCPP_STATIC_ASSERT(
        std::is_same_v<decltype(store.__args),
                       std::array<std::basic_format_arg<Context>, 2>>);
    LIBCPP_ASSERT(store.__args.size() == 2);
  }
  {
    auto store = std::make_format_args<Context>(1, 'c', nullptr);
    LIBCPP_STATIC_ASSERT(
        std::is_same_v<decltype(store),
                       std::__format_arg_store<Context, int, char, std::nullptr_t>>);
    LIBCPP_STATIC_ASSERT(
        std::is_same_v<decltype(store.__args),
                       std::array<std::basic_format_arg<Context>, 3>>);
    LIBCPP_ASSERT(store.__args.size() == 3);
  }
}

void test() {
  test<char>();
  test<wchar_t>();
}

int main(int, char**) {
  test();

  return 0;
}
