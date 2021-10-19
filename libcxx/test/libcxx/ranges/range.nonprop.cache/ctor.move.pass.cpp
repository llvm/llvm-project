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

// __non_propagating_cache(__non_propagating_cache&&);

#include <ranges>

#include <cassert>
#include <type_traits>
#include <utility>

template<bool NoexceptMove>
struct MoveConstructible {
  int x;
  constexpr explicit MoveConstructible(int i) : x(i) { }
  constexpr MoveConstructible(MoveConstructible&& other) noexcept(NoexceptMove) : x(other.x) { other.x = -1; }
  MoveConstructible& operator=(MoveConstructible&&) = default;
};

template <class T>
constexpr void test() {
  using Cache = std::ranges::__non_propagating_cache<T>;
  static_assert(std::is_nothrow_move_constructible_v<Cache>);

  // Test with direct initialization
  {
    Cache a;
    a.__emplace(3);

    Cache b(std::move(a));
    assert(!b.__has_value()); // make sure we don't propagate
    assert(!a.__has_value()); // make sure we disengage the source
  }

  // Test with copy initialization
  {
    Cache a;
    a.__emplace(3);

    Cache b = std::move(a);
    assert(!b.__has_value()); // make sure we don't propagate
    assert(!a.__has_value()); // make sure we disengage the source
  }
}

constexpr bool tests() {
  test<MoveConstructible<true>>();
  test<MoveConstructible<false>>();
  test<int>();
  return true;
}

int main(int, char**) {
  static_assert(tests());
  tests();
  return 0;
}
