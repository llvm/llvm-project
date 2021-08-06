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

// join_view() requires default_initializable<V> = default;

#include <cassert>
#include <ranges>

#include "test_macros.h"
#include "types.h"


constexpr bool test() {
  std::ranges::join_view<ParentView<ChildView>> jv;
  assert(std::move(jv).base().ptr_ == globalChildren);

  static_assert( std::default_initializable<std::ranges::join_view<ParentView<ChildView>>>);
  static_assert(!std::default_initializable<std::ranges::join_view<CopyableParent>>);

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
