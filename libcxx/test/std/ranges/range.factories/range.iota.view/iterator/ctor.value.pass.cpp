//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// constexpr explicit iterator(W value);

#include <ranges>
#include <cassert>

#include "test_macros.h"
#include "../types.h"

constexpr bool test() {
  {
    using Iter = std::ranges::iterator_t<std::ranges::iota_view<int>>;
    auto iter = Iter(42);
    assert(*iter == 42);
  }
  {
    using Iter = std::ranges::iterator_t<std::ranges::iota_view<SomeInt>>;
    auto iter = Iter(SomeInt(42));
    assert(*iter == SomeInt(42));
  }
  {
    using Iter = std::ranges::iterator_t<std::ranges::iota_view<SomeInt>>;
    static_assert(!std::is_convertible_v<Iter, SomeInt>);
    static_assert( std::is_constructible_v<Iter, SomeInt>);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
