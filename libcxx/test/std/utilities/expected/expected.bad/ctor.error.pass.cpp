//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// explicit bad_expected_access(E e);

// Effects: Initializes unex with std::move(e).

#include <cassert>
#include <concepts>
#include <expected>
#include <utility>

#include "test_macros.h"
#include "MoveOnly.h"

// test explicit
static_assert(std::convertible_to<int, int>);
static_assert(!std::convertible_to<int, std::bad_expected_access<int>>);

int main(int, char**) {
  std::bad_expected_access<MoveOnly> b(MoveOnly{3});
  assert(b.error().get() == 3);

  return 0;
}
