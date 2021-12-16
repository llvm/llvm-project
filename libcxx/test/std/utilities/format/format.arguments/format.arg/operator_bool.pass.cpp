//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: libcpp-has-no-incomplete-format
// TODO FMT Evaluate gcc-11 status
// UNSUPPORTED: gcc-11

// <format>

// explicit operator bool() const noexcept
//
// Note more testing is done in the unit test for:
// template<class Visitor, class Context>
//   see below visit_format_arg(Visitor&& vis, basic_format_arg<Context> arg);

#include <format>
#include <cassert>
#include <type_traits>

#include "test_macros.h"

void test(const auto& store) {
#ifdef _LIBCPP_VERSION
  for (const auto& arg : store.__args) {
    assert(arg);
    assert(static_cast<bool>(arg));
  }
#else
  (void)store;
#endif
}

template <class CharT>
void test() {
  using Context = std::basic_format_context<CharT*, CharT>;
  {
    std::basic_format_arg<Context> format_arg{};
    ASSERT_NOEXCEPT(!format_arg);
    assert(!format_arg);
    ASSERT_NOEXCEPT(static_cast<bool>(format_arg));
    assert(!static_cast<bool>(format_arg));
  }
  test(std::make_format_args<Context>());
  test(std::make_format_args<Context>(1));
  test(std::make_format_args<Context>(1, 'c'));
  test(std::make_format_args<Context>(1, 'c', nullptr));
}

void test() {
  test<char>();
  test<wchar_t>();
}

int main(int, char**) {
  test();

  return 0;
}
