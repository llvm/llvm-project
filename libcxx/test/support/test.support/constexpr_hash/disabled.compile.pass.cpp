//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// "support/constexpr_hash.h"

// Test that constexpr_hash<T> is disabled by default for some types like std::hash.

#include <type_traits>

#include "constexpr_hash.h"

struct X {};

template <class T>
void test() {
  using H = support::constexpr_hash<X>;
  static_assert(!std::is_default_constructible_v<H>);
  static_assert(!std::is_copy_constructible_v<H>);
  static_assert(!std::is_move_constructible_v<H>);
  static_assert(!std::is_copy_assignable_v<H>);
  static_assert(!std::is_move_assignable_v<H>);
  static_assert(!std::is_invocable_v<H, X&>);
  static_assert(!std::is_invocable_v<H, X const&>);
}

void test() { test<X>(); }
