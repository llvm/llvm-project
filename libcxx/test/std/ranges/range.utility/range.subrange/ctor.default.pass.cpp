//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// constexpr subrange() requires default_initializable<I>;

#include <ranges>

#include <cassert>
#include <cstddef>

#include "test_iterators.h"

// An input_or_output_iterator that is not default constructible so we can test
// the `requires` on subrange's default constructor.
struct NoDefaultIterator {
  using difference_type = std::ptrdiff_t;
  NoDefaultIterator() = delete;
  NoDefaultIterator& operator++();
  void operator++(int);
  int& operator*() const;
  friend bool operator==(NoDefaultIterator const&, NoDefaultIterator const&);
};
static_assert(std::input_or_output_iterator<NoDefaultIterator>);

// A sentinel type for the above iterator
struct Sentinel {
  friend bool operator==(NoDefaultIterator const&, Sentinel const&);
  friend bool operator==(Sentinel const&, NoDefaultIterator const&);
  friend bool operator!=(NoDefaultIterator const&, Sentinel const&);
  friend bool operator!=(Sentinel const&, NoDefaultIterator const&);
};

constexpr bool test() {
  {
    static_assert(!std::is_default_constructible_v<std::ranges::subrange<NoDefaultIterator, Sentinel, std::ranges::subrange_kind::sized>>);
    static_assert(!std::is_default_constructible_v<std::ranges::subrange<NoDefaultIterator, Sentinel, std::ranges::subrange_kind::unsized>>);
  }

  {
    using Iter = forward_iterator<int*>;
    std::ranges::subrange<Iter, Iter, std::ranges::subrange_kind::sized> subrange;
    assert(subrange.begin() == Iter());
    assert(subrange.end() == Iter());
  }
  {
    using Iter = forward_iterator<int*>;
    std::ranges::subrange<Iter, Iter, std::ranges::subrange_kind::unsized> subrange;
    assert(subrange.begin() == Iter());
    assert(subrange.end() == Iter());
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
