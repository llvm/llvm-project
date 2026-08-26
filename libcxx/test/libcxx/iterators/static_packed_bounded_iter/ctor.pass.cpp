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

#include <__iterator/static_packed_bounded_iter.h>
#include <cassert>
#include <cstddef>
#include <type_traits>

struct Base {
  long x = 0;
};

struct Derived : public Base {
  long y;

  constexpr Derived(long y) : Base(1), y(y) {}
};

template <typename T, std::size_t _Capacity>
using Iter = std::__static_packed_bounded_iterator<T, T, _Capacity>;

constexpr bool test() {
  {
    Derived a[1] = {{2}};
    auto it      = std::__make_static_packed_bounded_iter<Derived*, Derived*, 1>(a);

    assert(it->x == 1);
    assert(it->y == 2);

    static_assert(std::is_convertible_v<Derived*, Base*>);

    auto it2 = std::__static_packed_bounded_iterator<Base*, Derived*, 1>{it};
    assert(it2->x == 1);
  }

  static_assert(!std::is_constructible_v<Iter<Derived*, 1>, Iter<Base*, 1>>);
  static_assert(!std::is_constructible_v<Iter<int*, 1>, Iter<long*, 1>>);

  return true;
}

int main(int, char**) {
  assert(test());
  static_assert(test());

  return 0;
}
