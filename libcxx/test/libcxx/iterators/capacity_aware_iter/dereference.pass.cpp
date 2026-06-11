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

// operator[](difference_type)
// operator*();
// operator->();

#include <__iterator/capacity_aware_iterator.h>
#include <cassert>
#include <concepts>
#include <iterator>

#include "test_iterators.h"
#include "test_macros.h"

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
    std::same_as<Foo&> decltype(auto) res = it[0];
    ASSERT_NOEXCEPT(it[0]);
    assert(res == arr[0]);
    assert(&res == &arr[0]);
    assert(it[1] == arr[1]);
    assert(it[2] == arr[2]);

    CapIter it2 = it + 2;

    assert(it2[-1] == arr[1]);
    assert(it2[-2] == arr[0]);
  }

  // operator*
  {
    std::same_as<Foo&> decltype(auto) res = *it;
    ASSERT_NOEXCEPT(*it);
    assert(*it == arr[0]);
    assert(&res == &arr[0]);
    assert(&res == &(*it));
  }

  // operator->
  {
    std::same_as<Foo*> decltype(auto) ptr = it.operator->();
    ASSERT_NOEXCEPT(it->x);
    assert(ptr->x == 1);
    assert(ptr == &arr[0]);
  }

  return true;
}

int main(int, char**) {
  test<three_way_contiguous_iterator<Foo*>>();
  static_assert(test<three_way_contiguous_iterator<Foo*>>());

  return 0;
}
