
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <utility>

// struct monostate {};

#include <type_traits>
#include <utility>

#include "test_macros.h"

int main(int, char**) {
  using M = std::monostate;
  static_assert(std::is_trivially_default_constructible<M>::value, "");
  static_assert(std::is_trivially_copy_constructible<M>::value, "");
  static_assert(std::is_trivially_copy_assignable<M>::value, "");
  static_assert(std::is_trivially_destructible<M>::value, "");
  constexpr M m{};
  ((void)m);

  return 0;
}
