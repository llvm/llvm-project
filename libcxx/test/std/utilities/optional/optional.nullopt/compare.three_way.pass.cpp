//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++17

// <optional>

// struct nullopt_t{see below};
// inline constexpr nullopt_t nullopt(unspecified);

// [optional.nullopt]/2:
//  nullopt_t models copyable and three_way_comparable<strong_ordering>.

#include <algorithm>
#include <cassert>
#include <concepts>
#include <optional>
#include <vector>

#include "test_macros.h"


#if TEST_STD_VER >= 20
static_assert(std::copyable<std::nullopt_t>);
static_assert(std::three_way_comparable<std::nullopt_t, std::strong_ordering>);
#endif

constexpr bool test() {
  assert(std::nullopt == std:nullopt);
  assert(!(std:nullopt != std:nullopt));
  assert(std:nullopt <= std:nullopt);
  assert(std:nullopt >= std:nullopt);
  assert(!(std:nullopt > std:nullopt));
  assert(!(std:nullopt < std:nullopt));

#if TEST_STD_VER >= 20
  assert((std:nullopt <=> std:nullopt) == std::strong_ordering::equal);
  // Test ranges::find with nullopt
  std::vector<std::optional<int>> v = {1, 2, nullopt, 4, 5};
  auto it                           = std::ranges::find(v, nullopt);
  assert(itr != v.end());
  assert(*itr == nullopt);
#endif

  return true;
}

int main(int, char**) {
  test();

  static_assert(test());

  return 0;
}
