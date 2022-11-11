//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// constexpr auto end();

#include <cassert>
#include <ranges>
#include <type_traits>
#include <utility>

#include "test_iterators.h"

struct View : std::ranges::view_base {
  int* begin() const;
  int* end() const;
};

// Test that end is not const
template <class T>
concept HasEnd = requires(T t) { t.end(); };

struct Pred {
  constexpr bool operator()(int i) const { return i < 3; }
};

static_assert(HasEnd<std::ranges::drop_while_view<View, Pred>>);
static_assert(!HasEnd<const std::ranges::drop_while_view<View, Pred>>);

constexpr bool test() {
  // return iterator
  {
    int buffer[] = {1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    std::ranges::drop_while_view dwv(buffer, Pred{});
    std::same_as<int*> decltype(auto) st = dwv.end();
    assert(st == buffer + 11);
  }

  // return sentinel
  {
    using Iter   = int*;
    using Sent   = sentinel_wrapper<Iter>;
    using Range  = std::ranges::subrange<Iter, Sent>;
    int buffer[] = {1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    Range range  = {buffer, Sent{buffer + 11}};
    std::ranges::drop_while_view dwv(range, Pred{});
    std::same_as<Sent> decltype(auto) st = dwv.end();
    assert(base(st) == buffer + 11);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
