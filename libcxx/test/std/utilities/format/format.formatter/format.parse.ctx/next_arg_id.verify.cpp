//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// constexpr size_t next_arg_id()

#include <format>

constexpr bool test() {
  // [format.parse.ctx]/8
  // Let cur-arg-id be the value of next_arg_id_ prior to this call. Call
  // expressions where cur-arg-id >= num_args_ is true are not core constant
  // expressions (7.7 [expr.const]).
  std::format_parse_context context("", 0);
  context.next_arg_id();

  return true;
}

void f() {
  // expected-error@+1 {{static assertion expression is not an integral constant expression}}
  static_assert(test());
}
