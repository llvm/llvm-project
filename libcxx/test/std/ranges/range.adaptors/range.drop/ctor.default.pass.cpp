//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: gcc-10
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// drop_view() requires default_initializable<V> = default;

#include <ranges>

#include "test_macros.h"
#include "types.h"

constexpr bool test() {
  std::ranges::drop_view<ContiguousView> dropView1;
  assert(std::ranges::begin(dropView1) == globalBuff);

  static_assert( std::is_default_constructible_v<std::ranges::drop_view<ForwardView>>);
  static_assert(!std::is_default_constructible_v<std::ranges::drop_view<NoDefaultCtorForwardView>>);

  static_assert( std::is_nothrow_default_constructible_v<std::ranges::drop_view<ForwardView>>);
  static_assert(!std::is_nothrow_default_constructible_v<ThrowingDefaultCtorForwardView>);
  static_assert(!std::is_nothrow_default_constructible_v<std::ranges::drop_view<ThrowingDefaultCtorForwardView>>);

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
