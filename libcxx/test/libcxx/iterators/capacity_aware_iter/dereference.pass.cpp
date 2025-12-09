//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// template <class _Iterator, class _Container, class _ContainerMaxElements>
// struct __capacity_aware_iterator;

// Dereference operators

#include <__iterator/capacity_aware_iterator.h>

#include "test_iterators.h"

struct Foo {
  int x;
  constexpr bool operator==(Foo const& other) const { return x == other.x; }
};

template <typename Iter>
constexpr bool test() {
  Foo arr[]         = {Foo{1}, Foo{2}, Foo{3}, Foo{4}};
  constexpr long sz = std::size(arr);

  using CapIter = std::__capacity_aware_iterator<Iter, decltype(arr), sz>;

  CapIter it = std::__make_capacity_aware_iterator<Iter, decltype(arr), sz>(Iter(arr));

  // operator[]
  {
    assert(it[0] == Foo{1});
    assert(it[1] == Foo{2});
    assert(it[2] == Foo{3});

    CapIter it2 = it + 2;

    assert(it2[-1] == Foo{2});
    assert(it2[-2] == Foo{1});
  }

  // operator*
  {
    assert(*it == Foo{1});
  }

  // operator->
  {
    assert(it->x == 1);
  }

  return true;
}

int main(int, char**) {
  assert(test<three_way_contiguous_iterator<Foo*>>());
  static_assert(test<three_way_contiguous_iterator<Foo*>>());
}
