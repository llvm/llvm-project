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

// friend constexpr bool operator<(const iterator& x, const iterator& y)
//   requires totally_ordered<W>;
// friend constexpr bool operator>(const iterator& x, const iterator& y)
//   requires totally_ordered<W>;
// friend constexpr bool operator<=(const iterator& x, const iterator& y)
//   requires totally_ordered<W>;
// friend constexpr bool operator>=(const iterator& x, const iterator& y)
//   requires totally_ordered<W>;
// friend constexpr bool operator==(const iterator& x, const iterator& y)
//   requires equality_comparable<W>;

// TODO: test spaceship operator once it's implemented.

#include <ranges>
#include <cassert>

#include "test_macros.h"
#include "../types.h"

constexpr bool test() {
  {
    const std::ranges::iota_view<int> io(0);
    assert(                  io.begin()  ==                   io.begin() );
    assert(                  io.begin()  != std::ranges::next(io.begin()));
    assert(                  io.begin()  <  std::ranges::next(io.begin()));
    assert(std::ranges::next(io.begin()) >                    io.begin() );
    assert(                  io.begin()  <= std::ranges::next(io.begin()));
    assert(std::ranges::next(io.begin()) >=                   io.begin() );
    assert(                  io.begin()  <=                   io.begin() );
    assert(                  io.begin()  >=                   io.begin() );
  }
  {
    std::ranges::iota_view<int> io(0);
    assert(                  io.begin()  ==                   io.begin() );
    assert(                  io.begin()  != std::ranges::next(io.begin()));
    assert(                  io.begin()  <  std::ranges::next(io.begin()));
    assert(std::ranges::next(io.begin()) >                    io.begin() );
    assert(                  io.begin()  <= std::ranges::next(io.begin()));
    assert(std::ranges::next(io.begin()) >=                   io.begin() );
    assert(                  io.begin()  <=                   io.begin() );
    assert(                  io.begin()  >=                   io.begin() );
  }
  {
    const std::ranges::iota_view<SomeInt> io(SomeInt(0));
    assert(                  io.begin()  ==                   io.begin() );
    assert(                  io.begin()  != std::ranges::next(io.begin()));
    assert(                  io.begin()  <  std::ranges::next(io.begin()));
    assert(std::ranges::next(io.begin()) >                    io.begin() );
    assert(                  io.begin()  <= std::ranges::next(io.begin()));
    assert(std::ranges::next(io.begin()) >=                   io.begin() );
    assert(                  io.begin()  <=                   io.begin() );
    assert(                  io.begin()  >=                   io.begin() );
  }
  {
    std::ranges::iota_view<SomeInt> io(SomeInt(0));
    assert(                  io.begin()  ==                   io.begin() );
    assert(                  io.begin()  != std::ranges::next(io.begin()));
    assert(                  io.begin()  <  std::ranges::next(io.begin()));
    assert(std::ranges::next(io.begin()) >                    io.begin() );
    assert(                  io.begin()  <= std::ranges::next(io.begin()));
    assert(std::ranges::next(io.begin()) >=                   io.begin() );
    assert(                  io.begin()  <=                   io.begin() );
    assert(                  io.begin()  >=                   io.begin() );
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
