//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// take_view() requires default_initializable<V> = default;

#include <ranges>
#include <cassert>

int buff[8] = {1, 2, 3, 4, 5, 6, 7, 8};

struct DefaultConstructible : std::ranges::view_base {
  constexpr DefaultConstructible() : begin_(buff), end_(buff + 8) { }
  constexpr int const* begin() const { return begin_; }
  constexpr int const* end() const { return end_; }
private:
  int const* begin_;
  int const* end_;
};

struct NonDefaultConstructible : std::ranges::view_base {
  NonDefaultConstructible() = delete;
  int* begin() const;
  int* end() const;
};

constexpr bool test() {
  {
    std::ranges::take_view<DefaultConstructible> tv;
    assert(tv.begin() == buff);
    assert(tv.size() == 0);
  }

  // Test SFINAE-friendliness
  {
    static_assert( std::is_default_constructible_v<std::ranges::take_view<DefaultConstructible>>);
    static_assert(!std::is_default_constructible_v<std::ranges::take_view<NonDefaultConstructible>>);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
