//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr void value() &&;
// Mandates: is_copy_constructible_v<E> is true and is_move_constructible_v<E> is true.

#include <expected>

#include "MoveOnly.h"

void test() {
  // MoveOnly
  std::expected<void, MoveOnly> e(std::unexpect, 5);

  // COMPILE FAIL: MoveOnly is not copy constructible
  std::move(e).value();
}

int main(int, char**) { test(); }
