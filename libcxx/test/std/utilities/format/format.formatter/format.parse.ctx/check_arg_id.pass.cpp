//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: no-exceptions
// UNSUPPORTED: libcpp-has-no-incomplete-format

// This test requires the dylib support introduced in D92214.
// XFAIL: use_system_cxx_lib && target={{.+}}-apple-macosx10.{{9|10|11|12|13|14|15}}

// <format>

// constexpr void check_arg_id(size_t id);

#include <format>

#include <cassert>
#include <cstring>
#include <string_view>

#include "test_macros.h"

constexpr bool test() {
  std::format_parse_context context("", 10);
  for (size_t i = 0; i < 10; ++i)
    context.check_arg_id(i);

  return true;
}

void test_exception() {
  [] {
    std::format_parse_context context("", 1);
    TEST_IGNORE_NODISCARD context.next_arg_id();
    try {
      context.check_arg_id(0);
      assert(false);
    } catch ([[maybe_unused]] const std::format_error& e) {
      LIBCPP_ASSERT(std::strcmp(e.what(), "Using manual argument numbering in automatic argument numbering mode") == 0);
      return;
    }
    assert(false);
  }();

  auto test_arg = [](size_t num_args) {
    std::format_parse_context context("", num_args);
    // Out of bounds access is valid if !std::is_constant_evaluated()
    for (size_t i = 0; i <= num_args; ++i)
      context.check_arg_id(i);
  };
  for (size_t i = 0; i < 10; ++i)
    test_arg(i);
}

int main(int, char**) {
  test();
  test_exception();
  static_assert(test());

  return 0;
}
