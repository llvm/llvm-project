//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// template <class _Ptr, class _Tag, size_t _RangeCapacity>
// class __static_packed_bounded_iter;
//
// Comparison operators

#include <__iterator/static_packed_bounded_iter.h>
#include <cassert>
#include <compare>
#include <concepts>
#include <iterator>

#include "test_macros.h"

struct alignas(4) Foo {
  int x;
  constexpr Foo(int y) : x(y) {}
};

template <class Iter>
constexpr bool tests() {
  Foo array[]             = {0, 1};
  Foo* b                  = array + 0;
  using BoundedIter       = std::__static_packed_bounded_iterator<Iter, decltype(array), std::size(array)>;
  BoundedIter const iter1 = std::__make_static_packed_bounded_iter<Iter, decltype(array), std::size(array)>(Iter(b));
  BoundedIter const iter2 =
      std::__make_static_packed_bounded_iter<Iter, decltype(array), std::size(array)>(Iter(b)) + 2;

  // operator==
  {
    assert(iter1 == iter1);
    assert(!(iter1 == iter2));
  }
  // operator!=
  {
    assert(iter1 != iter2);
    assert(!(iter1 != iter1));
  }
  // operator<
  {
    assert(iter1 < iter2);
    assert(!(iter2 < iter1));
    assert(!(iter1 < iter1));
  }
  // operator>
  {
    assert(iter2 > iter1);
    assert(!(iter1 > iter2));
    assert(!(iter1 > iter1));
  }
  // operator<=
  {
    assert(iter1 <= iter2);
    assert(!(iter2 <= iter1));
    assert(iter1 <= iter1);
  }
  // operator>=
  {
    assert(iter2 >= iter1);
    assert(!(iter1 >= iter2));
    assert(iter1 >= iter1);
  }

  std::same_as<std::strong_ordering> decltype(auto) r1 = iter1 <=> iter2;
  assert(r1 == std::strong_ordering::less);

  return true;
}

int main(int, char**) {
  tests<Foo*>();
  static_assert(tests<Foo*>(), "");

  return 0;
}
