//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// "support/constexpr_hash.h"

#include "constexpr_hash.h"

#include <cassert>
#include <cstddef>
#include <type_traits>

#include "test_macros.h"

constexpr void test_nullptr() {
  typedef std::nullptr_t T;
  using H = support::constexpr_hash<T>;

  ASSERT_NOEXCEPT(H()(T()));
}

constexpr bool test_with_hash() {
  test_nullptr();
  return true;
}

int main(int, char**) {
  test_with_hash();
  static_assert(test_with_hash());

  return 0;
}
