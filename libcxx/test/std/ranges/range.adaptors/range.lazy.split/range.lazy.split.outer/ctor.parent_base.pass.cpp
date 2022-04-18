//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// constexpr outer-iterator(Parent& parent, iterator_t<Base> current);
//   requires forward_range<Base>

#include <ranges>

#include <type_traits>
#include <utility>
#include "../types.h"

static_assert(!std::ranges::forward_range<SplitViewInput>);
static_assert(!std::is_constructible_v<OuterIterInput, SplitViewInput&, std::ranges::iterator_t<InputView>>);

constexpr bool test() {
  ForwardView input("abc");
  SplitViewForward v(std::move(input), " ");
  [[maybe_unused]] OuterIterForward i(v, input.begin());

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
