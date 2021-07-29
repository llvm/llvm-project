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

// __non_propagating_cache();

#include <ranges>

#include <cassert>
#include <type_traits>

struct HasDefault { HasDefault() = default; };
struct NoDefault { NoDefault() = delete; };

template <class T>
constexpr void test() {
  using Cache = std::ranges::__non_propagating_cache<T>;
  static_assert(std::is_nothrow_default_constructible_v<Cache>);
  Cache cache;
  assert(!cache.__has_value());
}

constexpr bool tests() {
  test<HasDefault>();
  test<NoDefault>();
  test<int>();
  test<char*>();
  return true;
}

int main(int, char**) {
  static_assert(tests());
  tests();
  return 0;
}
