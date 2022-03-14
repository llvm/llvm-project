//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-format
// XFAIL: libcpp-has-no-wide-characters
// TODO FMT Evaluate gcc-11 status
// UNSUPPORTED: gcc-11

// <format>

// template<class... Args>
//   format-arg-store<wformat_context, Args...>
//   make_wformat_args(const Args&... args);

#include <cassert>
#include <format>

#include "test_basic_format_arg.h"
#include "test_macros.h"

int main(int, char**) {
  using Context [[maybe_unused]] = std::basic_format_context<
      std::back_insert_iterator<std::basic_string<wchar_t>>, wchar_t>;

  [[maybe_unused]] auto value = std::make_wformat_args(42, nullptr, false, 1.0);

  LIBCPP_ASSERT(value.__args.size() == 4);
  LIBCPP_ASSERT(test_basic_format_arg(value.__args[0], 42));
  // Note [format.arg]/11 specifies a nullptr is stored as a const void*.
  LIBCPP_ASSERT(test_basic_format_arg(value.__args[1],
                                      static_cast<const void*>(nullptr)));
  LIBCPP_ASSERT(test_basic_format_arg(value.__args[2], false));
  LIBCPP_ASSERT(test_basic_format_arg(value.__args[3], 1.0));

  return 0;
}
