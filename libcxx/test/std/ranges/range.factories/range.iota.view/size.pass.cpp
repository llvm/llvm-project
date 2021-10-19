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

// constexpr auto size() const requires see below;

#include <ranges>
#include <cassert>
#include <limits>

#include "test_macros.h"
#include "types.h"

constexpr bool test() {
  // Both are integer like and both are less than zero.
  {
    const std::ranges::iota_view<int, int> io(-10, -5);
    assert(io.size() == 5);
  }
  {
    const std::ranges::iota_view<int, int> io(-10, -10);
    assert(io.size() == 0);
  }

  // Both are integer like and "value_" is less than zero.
  {
    const std::ranges::iota_view<int, int> io(-10, 10);
    assert(io.size() == 20);
  }
  {
// TODO: this is invalid with the current implementation. We need to file an LWG issue to
// fix this. Essentially the issue is: An int's min and max are -2147483648 and 2147483647
// which means the negated min cannot be represented as an integer; it needs to be cast to
// an unsigned type first. That seems to be what the
// to-unsigned-like(bound_) + to-unsigned-like(-value_))
// part of https://eel.is/c++draft/range.iota#view-15 is doing, but I think it's doing it
// wrong. It should be to-unsigned-like(bound_) - to-unsigned-like(value_)) (cast to
// unsigned first).
//     const std::ranges::iota_view<int, int> io(std::numeric_limits<int>::min(), std::numeric_limits<int>::max());
//     assert(io.size() == (static_cast<unsigned>(std::numeric_limits<int>::max()) * 2) + 1);
  }

  // It is UB for "bound_" to be less than "value_" i.e.: iota_view<int, int> io(10, -5).

  // Both are integer like and neither less than zero.
  {
    const std::ranges::iota_view<int, int> io(10, 20);
    assert(io.size() == 10);
  }
  {
    const std::ranges::iota_view<int, int> io(10, 10);
    assert(io.size() == 0);
  }
  {
    const std::ranges::iota_view<int, int> io(0, 0);
    assert(io.size() == 0);
  }
  {
    const std::ranges::iota_view<int, int> io(0, std::numeric_limits<int>::max());
    assert(io.size() == std::numeric_limits<int>::max());
  }

  // Neither are integer like.
  {
    const std::ranges::iota_view<SomeInt, SomeInt> io(SomeInt(-20), SomeInt(-10));
    assert(io.size() == 10);
  }
  {
    const std::ranges::iota_view<SomeInt, SomeInt> io(SomeInt(-10), SomeInt(-10));
    assert(io.size() == 0);
  }
  {
    const std::ranges::iota_view<SomeInt, SomeInt> io(SomeInt(0), SomeInt(0));
    assert(io.size() == 0);
  }
  {
    const std::ranges::iota_view<SomeInt, SomeInt> io(SomeInt(10), SomeInt(20));
    assert(io.size() == 10);
  }
  {
    const std::ranges::iota_view<SomeInt, SomeInt> io(SomeInt(10), SomeInt(10));
    assert(io.size() == 0);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
