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
//
// std::pointer_traits specialization

#include <__iterator/capacity_aware_iterator.h>
#include <cassert>
#include <cstddef>
#include <iterator>
#include <memory>
#include <type_traits>

#include "test_iterators.h"
#include "test_macros.h"

template <class Iter>
TEST_CONSTEXPR_CXX14 bool tests() {
  int array[] = {0, 1, 2, 3, 4};

  using CapIter           = std::__capacity_aware_iterator<Iter, decltype(array), std::size(array)>;
  using PointerTraits     = std::pointer_traits<CapIter>;
  using BasePointerTraits = std::pointer_traits<Iter>;
  static_assert(std::is_same_v<typename PointerTraits::pointer, CapIter>);
  static_assert(std::is_same_v<typename PointerTraits::element_type, typename BasePointerTraits::element_type>);
  static_assert(std::is_same_v<typename PointerTraits::difference_type, typename BasePointerTraits::difference_type>);

  {
    int* b              = array + 0;
    int* e              = array + 5;
    CapIter const iter1 = std::__make_capacity_aware_iterator<Iter, decltype(array), std::size(array)>(Iter(b));
    CapIter const iter2 = std::__make_capacity_aware_iterator<Iter, decltype(array), std::size(array)>(Iter(e));
    assert(std::to_address(iter1) == b); // in-bounds iterator
    assert(std::to_address(iter2) == e); // out-of-bounds iterator
  }

  return true;
}

int main(int, char**) {
  tests<int*>();
  static_assert(tests<int*>(), "");
  tests<contiguous_iterator<int*> >();
  static_assert(tests<contiguous_iterator<int*> >(), "");

  return 0;
}
