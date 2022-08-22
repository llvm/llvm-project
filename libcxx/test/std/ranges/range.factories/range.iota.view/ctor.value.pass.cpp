//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// constexpr explicit iota_view(W value);

#include <ranges>
#include <cassert>

#include "test_macros.h"
#include "types.h"

struct SomeIntComparable {
  using difference_type = int;

  SomeInt value_;
  constexpr SomeIntComparable() : value_(SomeInt(10)) {}

  friend constexpr bool operator==(SomeIntComparable lhs, SomeIntComparable rhs) {
    return lhs.value_ == rhs.value_;
  }
  friend constexpr bool operator==(SomeIntComparable lhs, SomeInt rhs) {
    return lhs.value_ == rhs;
  }
  friend constexpr bool operator==(SomeInt lhs, SomeIntComparable rhs) {
    return lhs == rhs.value_;
  }

  friend constexpr difference_type operator-(SomeIntComparable lhs, SomeIntComparable rhs) {
    return lhs.value_ - rhs.value_;
  }

  constexpr SomeIntComparable& operator++() { ++value_; return *this; }
  constexpr SomeIntComparable  operator++(int) { auto tmp = *this; ++value_; return tmp; }
  constexpr SomeIntComparable  operator--() { --value_; return *this; }
};

constexpr bool test() {
  {
    std::ranges::iota_view<SomeInt> io(SomeInt(42));
    assert((*io.begin()).value_ == 42);
    // Check that end returns std::unreachable_sentinel.
    assert(io.end() != io.begin());
    static_assert(std::same_as<decltype(io.end()), std::unreachable_sentinel_t>);
  }

  {
    std::ranges::iota_view<SomeInt, SomeIntComparable> io(SomeInt(0));
    assert(std::ranges::next(io.begin(), 10) == io.end());
  }
  {
    static_assert(!std::is_convertible_v<std::ranges::iota_view<SomeInt>, SomeInt>);
    static_assert( std::is_constructible_v<std::ranges::iota_view<SomeInt>, SomeInt>);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
