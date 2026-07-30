//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// template <class _Ptr, class _Tag, size_t _RangeCapacity>
// class __static_packed_bounded_iter;
//
// Dereference and indexing operators

// REQUIRES: std-at-least-c++26

#include <__iterator/static_packed_bounded_iter.h>
#include <cassert>
#include <iterator>

#include "test_macros.h"

struct alignas(8) Foo {
  char x;
  constexpr bool operator==(Foo const& other) const { return x == other.x; }
};

template <class Iter>
TEST_CONSTEXPR_CXX14 bool tests() {
  Foo array[] = {Foo{40}, Foo{41}, Foo{42}, Foo{43}, Foo{44}};
  Foo* b      = array + 0;

  using BoundedIter = std::__static_packed_bounded_iterator<Foo*, decltype(array), std::size(array)>;

  BoundedIter const iter1 = std::__make_static_packed_bounded_iter<Foo*, decltype(array), std::size(array)>(Iter(b));
  BoundedIter const iter2 =
      std::__make_static_packed_bounded_iter<Foo*, decltype(array), std::size(array)>(Iter(b)) + 5;

  // operator*
  assert(*iter1 == Foo{40});
  // operator->
  assert(iter1->x == 40);
  // operator[]
  assert(iter1[0] == Foo{40});
  assert(iter1[1] == Foo{41});
  assert(iter1[2] == Foo{42});
  assert(iter2[-1] == Foo{44});
  assert(iter2[-2] == Foo{43});

  return true;
}

int main(int, char**) {
  tests<Foo*>();
  static_assert(tests<Foo*>(), "");

  return 0;
}
