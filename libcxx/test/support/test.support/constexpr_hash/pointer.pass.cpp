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

constexpr bool test() {
  using H = support::constexpr_hash<std::nullptr_t>;
  ASSERT_NOEXCEPT(H()(nullptr));
  return true;
}
int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
