//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <format>

// template<class... Args>
//   basic_format_args(const format-arg-store<Context, Args...>& store) noexcept;

#include <format>
#include <cassert>
#include <type_traits>

#include "test_macros.h"

template <class CharT>
void test() {
  int i         = 1;
  char c        = 'c';
  nullptr_t p   = nullptr;
  using Context = std::basic_format_context<CharT*, CharT>;
  static_assert(!std::is_default_constructible_v<std::basic_format_args<Context>>);
  {
    auto store = std::make_format_args<Context>(i);
    ASSERT_NOEXCEPT(std::basic_format_args<Context>{store});
    std::basic_format_args<Context> format_args{store};
    assert(format_args.get(0));
    assert(!format_args.get(1));
  }
  {
    auto store = std::make_format_args<Context>(i, c);
    ASSERT_NOEXCEPT(std::basic_format_args<Context>{store});
    std::basic_format_args<Context> format_args{store};
    assert(format_args.get(0));
    assert(format_args.get(1));
    assert(!format_args.get(2));
  }
  {
    auto store = std::make_format_args<Context>(i, c, p);
    ASSERT_NOEXCEPT(std::basic_format_args<Context>{store});
    std::basic_format_args<Context> format_args{store};
    assert(format_args.get(0));
    assert(format_args.get(1));
    assert(format_args.get(2));
    assert(!format_args.get(3));
  }
}

void test() {
  test<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif
}

int main(int, char**) {
  test();

  return 0;
}
