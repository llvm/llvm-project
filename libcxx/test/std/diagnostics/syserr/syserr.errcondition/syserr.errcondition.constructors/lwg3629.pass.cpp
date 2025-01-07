//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <system_error>

// class error_condition

// template <ErrorCodeEnum E> error_condition(E e);

// Regression test for https://github.com/llvm/llvm-project/issues/57614

int make_error_condition; // It's important that this comes before <system_error>

#include <system_error>
#include <cassert>
#include <type_traits>

namespace User {
  enum Err {};

  std::error_condition make_error_condition(Err) { return std::error_condition(42, std::generic_category()); }
}

template <>
struct std::is_error_condition_enum<User::Err> : true_type {};

int main(int, char**) {
  std::error_condition e((User::Err()));
  assert(e.value() == 42);
  assert(e.category() == std::generic_category());

  return 0;
}
