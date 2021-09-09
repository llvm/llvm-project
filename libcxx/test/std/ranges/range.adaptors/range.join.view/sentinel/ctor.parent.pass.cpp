//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// constexpr explicit sentinel(Parent& parent);

#include <cassert>
#include <ranges>

#include "test_macros.h"
#include "../types.h"

constexpr bool test() {
  int buffer[4][4] = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}};

  CopyableChild children[4] = {CopyableChild(buffer[0]), CopyableChild(buffer[1]), CopyableChild(buffer[2]), CopyableChild(buffer[3])};
  CopyableParent parent{children};
  std::ranges::join_view jv(parent);
  std::ranges::sentinel_t<decltype(jv)> sent(jv);
  assert(sent == std::ranges::next(jv.begin(), 16));

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  {
    // Test explicitness.
    using Parent = std::ranges::join_view<ParentView<ChildView>>;
    static_assert( std::is_constructible_v<std::ranges::sentinel_t<Parent>, Parent&>);
    static_assert(!std::is_convertible_v<std::ranges::sentinel_t<Parent>, Parent&>);
  }

  return 0;
}
