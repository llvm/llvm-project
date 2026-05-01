//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// iterator() = default;

#include <cassert>
#include <ranges>

#include "../../range_adaptor_types.h"

// A default-constructible forward iterator (required for non-first ranges in cartesian_product_view).
struct PODIter {
  int i; // deliberately uninitialised

  using iterator_category = std::random_access_iterator_tag;
  using value_type        = int;
  using difference_type   = std::intptr_t;

  constexpr int operator*() const { return i; }

  constexpr PODIter& operator++() { return *this; }
  constexpr PODIter operator++(int) { return *this; }

  friend constexpr bool operator==(const PODIter&, const PODIter&) = default;
};

struct IterDefaultCtrView : std::ranges::view_base {
  PODIter begin() const;
  PODIter end() const;
};

// `cartesian_product_view` requires all-but-first ranges to be `forward_range`, and
// `forward_iterator` is required to be `default_initializable`. So the only place a
// non-default-initializable underlying iterator may live is the first (input) range.
struct InputNoDefaultCtrView : std::ranges::view_base {
  cpp20_input_iterator<int*> begin() const;
  sentinel_wrapper<cpp20_input_iterator<int*>> end() const;
};

template <class... Views>
using cp_iter = std::ranges::iterator_t<std::ranges::cartesian_product_view<Views...>>;

static_assert(!std::default_initializable<cp_iter<InputNoDefaultCtrView>>);
static_assert(!std::default_initializable<cp_iter<InputNoDefaultCtrView, IterDefaultCtrView>>);
static_assert(std::default_initializable<cp_iter<IterDefaultCtrView>>);
static_assert(std::default_initializable<cp_iter<IterDefaultCtrView, IterDefaultCtrView>>);
static_assert(std::default_initializable<cp_iter<IterDefaultCtrView, IterDefaultCtrView, IterDefaultCtrView>>);

constexpr bool test() {
  using Iter = cp_iter<IterDefaultCtrView, IterDefaultCtrView>;
  {
    Iter it;
    (void)*it; // default-constructible and dereferenceable
  }
  {
    Iter it     = {};
    auto [x, y] = *it;
    assert(x == 0);
    assert(y == 0);
  }
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
