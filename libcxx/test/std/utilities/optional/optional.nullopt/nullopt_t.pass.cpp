//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// <optional>

// struct nullopt_t{see below};
// inline constexpr nullopt_t nullopt(unspecified);

// [optional.nullopt]/2:
//   Type nullopt_t does not have a default constructor or an initializer-list
//   constructor, and is not an aggregate. nullopt_t models copyable and
//   three_way_comparable<strong_ordering>.

#include <optional>
#include <type_traits>
#if _LIBCPP_STD_VER >= 26
#  include <vector>
#  include <ranges>
#  include <cassert>
#  include <algorithm>
#endif

#include "test_macros.h"

using std::nullopt;
using std::nullopt_t;

constexpr bool test() {
  nullopt_t foo{nullopt};
  (void)foo;
  return true;
}

int main(int, char**) {
  static_assert(std::is_empty_v<nullopt_t>);
  static_assert(!std::is_default_constructible_v<nullopt_t>);

  static_assert(std::is_same_v<const nullopt_t, decltype(nullopt)>);
  static_assert(test());
#if TEST_STD_VER >= 26
  // Test comparisons between nullopt_t
  static_assert(nullopt == nullopt);
  static_assert(!(nullopt != nullopt));
  static_assert(nullopt <= nullopt);
  static_assert(nullopt >= nullopt);
  static_assert((nullopt <=> nullopt) == std::strong_ordering::equal);

  // Test ranges::find with nullopt
  std::vector<std::optional<int>> v = {1, 2, nullopt, 4, 5};
  auto itr                          = std::ranges::find(v, nullopt);
  assert(itr != v.end());
  assert(*itr == nullopt);
#endif

  return 0;
}
