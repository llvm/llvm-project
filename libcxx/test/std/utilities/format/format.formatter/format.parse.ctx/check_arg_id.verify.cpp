//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-format

// constexpr void check_arg_id(size_t id);

#include <format>

#include "test_macros.h"

constexpr bool test() {
  // [format.parse.ctx]/11
  // Remarks: Call expressions where id >= num_args_ are not
  // core constant expressions ([expr.const]).
  std::format_parse_context context("", 0);
  context.check_arg_id(1);

  return true;
}

int main(int, char**) {
  // expected-error@+1 {{static_assert expression is not an integral constant expression}}
  static_assert(test());

  return 0;
}
