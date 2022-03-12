//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// constexpr bool __has_value() const;

#include <ranges>

#include <cassert>

template <class T>
constexpr void test() {
  using Cache = std::ranges::__non_propagating_cache<T>;

  // __has_value on an empty cache
  {
    Cache const cache;
    assert(!cache.__has_value());
  }

  // __has_value on a non-empty cache
  {
    Cache cache; cache.__emplace();
    assert(cache.__has_value());
  }
}

struct T { };

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
