//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// iterator() requires default_initializable<iterator_t<Base>> = default;

#include <ranges>
#include <tuple>

#include "../types.h"

struct PODIter : IterBase<PODIter> {
  int i; // deliberately uninitialised
};

struct IterDefaultCtrView : std::ranges::view_base {
  PODIter begin() const;
  PODIter end() const;
};

struct IterNoDefaultCtrView : std::ranges::view_base {
  cpp20_input_iterator<std::tuple<int>*> begin() const;
  sentinel_wrapper<cpp20_input_iterator<std::tuple<int>*>> end() const;
};

template <class View, size_t N>
using ElementsIter = std::ranges::iterator_t<std::ranges::elements_view<View, N>>;

static_assert(!std::default_initializable<ElementsIter<IterNoDefaultCtrView, 0>>);
static_assert(std::default_initializable<ElementsIter<IterDefaultCtrView, 0>>);

constexpr bool test() {
  using Iter = ElementsIter<IterDefaultCtrView, 0>;
  {
    Iter iter;
    assert(iter.base().i == 0); // PODIter has to be initialised to have value 0
  }

  {
    Iter iter = {};
    assert(iter.base().i == 0); // PODIter has to be initialised to have value 0
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
