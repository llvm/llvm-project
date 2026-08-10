//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// constexpr T const& operator*() const;
// constexpr T& operator*();

#include <ranges>

#include <cassert>

template <class T>
constexpr void test() {
  using Cache = std::ranges::__non_propagating_cache<T>;

  // non-const version
  {
    Cache cache; cache.__emplace(3);
    T& result = *cache;
    assert(result == T{3});
  }

  // const version
  {
    Cache cache; cache.__emplace(3);
    T const& result = *static_cast<Cache const&>(cache);
    assert(result == T{3});
  }
}

struct T {
  int x;
  constexpr explicit T(int i) : x(i) { }
  constexpr bool operator==(T const& other) const { return x == other.x; }
};

constexpr bool tests() {
  test<T>();
  test<int>();
  return true;
}

int main(int, char**) {
  static_assert(tests());
  tests();
  return 0;
}
