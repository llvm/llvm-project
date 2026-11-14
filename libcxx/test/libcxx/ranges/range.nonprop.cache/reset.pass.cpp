//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// constexpr void __reset();

#include <ranges>

#include <cassert>

template <class T>
constexpr void test() {
  using Cache = std::ranges::__non_propagating_cache<T>;

  // __reset on an empty cache
  {
    Cache cache;
    assert(!cache.__has_value());
    cache.__reset();
    assert(!cache.__has_value());
  }

  // __reset on a non-empty cache
  {
    Cache cache;
    cache.__emplace();
    assert(cache.__has_value());
    cache.__reset();
    assert(!cache.__has_value());
  }
}

struct T {};

constexpr bool tests() {
  test<T>();
  test<int>();
  return true;
}

int main(int, char**) {
  tests();
  static_assert(tests());

  return 0;
}
