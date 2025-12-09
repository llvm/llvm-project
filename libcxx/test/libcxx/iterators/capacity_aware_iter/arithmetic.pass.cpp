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

// Arithmetic operators

#include <__iterator/capacity_aware_iterator.h>
#include <cstddef>
#include <iterator>

#include "test_iterators.h"

template <typename Iter>
constexpr bool test() {
  int arr[]           = {1, 2, 3, 4, 5, 6};
  constexpr size_t sz = std::size(arr);

  using CapIter = std::__capacity_aware_iterator<Iter, decltype(arr), sz>;

  int* i = arr + 0;

  // operator++()
  {
    CapIter iter = std::__make_capacity_aware_iterator<Iter, decltype(arr), sz>(Iter(i));
    CapIter& res = ++iter;

    assert(&res == &iter);
    assert(*iter == 2);
  }

  // operator++(int)
  {
    CapIter iter = std::__make_capacity_aware_iterator<Iter, decltype(arr), sz>(Iter(i));
    CapIter res  = iter++;

    assert(*res == 1);
    assert(*iter == 2);
  }

  // operator--()
  {
    CapIter iter = std::__make_capacity_aware_iterator<Iter, decltype(arr), sz>(Iter(i + 1));
    CapIter& res = --iter;

    assert(&iter == &res);
    assert(*iter == 1);
  }

  // operator--(int)
  {
    CapIter iter = std::__make_capacity_aware_iterator<Iter, decltype(arr), sz>(Iter(i + 1));
    CapIter res  = iter--;

    assert(*res == 2);
    assert(*iter == 1);
  }

  // operator+=(difference_type)
  {
    CapIter iter = std::__make_capacity_aware_iterator<Iter, decltype(arr), sz>(Iter(i));
    CapIter& res = iter += 2;

    assert(&iter == &res);
    assert(*iter == 3);
  }

  // operator+(__capacity_aware_iterator, difference_type)
  {
    CapIter iter = std::__make_capacity_aware_iterator<Iter, decltype(arr), sz>(Iter(i));
    CapIter res  = iter + 2;

    assert(*iter == 1);
    assert(*res == 3);
  }

  // operator+(difference_type, __capacity_aware_iterator)
  {
    CapIter iter = std::__make_capacity_aware_iterator<Iter, decltype(arr), sz>(Iter(i));
    CapIter res  = 2 + iter;

    assert(*iter == 1);
    assert(*res == 3);
  }

  // operator-=(difference_type)
  {
    CapIter iter = std::__make_capacity_aware_iterator<Iter, decltype(arr), sz>(Iter(i + 2));
    CapIter& res = iter -= 2;

    assert(&iter == &res);
    assert(*iter == 1);
  }

  // operator-(__capacity_aware_iterator, difference_type)
  {
    CapIter iter = std::__make_capacity_aware_iterator<Iter, decltype(arr), sz>(Iter(i + 2));
    CapIter res  = iter - 2;

    assert(*iter == 3);
    assert(*res == 1);
  }

  // operator-(__capacity_aware_iterator, __capacity_aware_iterator)
  {
    CapIter iter        = std::__make_capacity_aware_iterator<Iter, decltype(arr), sz>(Iter(i));
    CapIter iter2       = std::__make_capacity_aware_iterator<Iter, decltype(arr), sz>(Iter(i + 2));
    CapIter iter3       = std::__make_capacity_aware_iterator<Iter, decltype(arr), sz>(Iter(i + 6));
    std::ptrdiff_t res  = iter2 - iter;
    std::ptrdiff_t res2 = iter3 - iter;

    assert(res == 2);
    assert(res2 == 6);
  }

  return true;
}

int main(int, char**) {
  assert(test<cpp20_random_access_iterator<int*>>());
  static_assert(test<cpp20_random_access_iterator<int*>>());

  return 0;
}
