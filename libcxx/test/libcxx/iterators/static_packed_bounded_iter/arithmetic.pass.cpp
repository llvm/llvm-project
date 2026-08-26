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
// Arithmetic operators

#include <__iterator/static_packed_bounded_iter.h>
#include <cassert>
#include <cstddef>
#include <iterator>

struct alignas(8) Foo {
  int x;

  constexpr Foo(int y) : x(y) {}

  constexpr bool operator==(const Foo& rhs) const { return x == rhs.x; }
};

template <class Iter>
constexpr bool tests() {
  Foo array[]       = {40, 41, 42, 43, 44};
  Foo* b            = array + 0;
  Foo* e            = array + 5;
  using BoundedIter = std::__static_packed_bounded_iterator<Iter, decltype(array), std::size(array)>;
  // ++it
  {
    BoundedIter iter    = std::__make_static_packed_bounded_iter<Iter, decltype(array), std::size(array)>(Iter(b));
    BoundedIter& result = ++iter;
    assert(&result == &iter);
    assert(*iter == 41);
  }
  // it++
  {
    BoundedIter iter   = std::__make_static_packed_bounded_iter<Iter, decltype(array), std::size(array)>(Iter(b));
    BoundedIter result = iter++;
    assert(*result == 40);
    assert(*iter == 41);
  }
  // --it
  {
    BoundedIter iter    = std::__make_static_packed_bounded_iter<Iter, decltype(array), std::size(array)>(Iter(b)) + 3;
    BoundedIter& result = --iter;
    assert(&result == &iter);
    assert(*iter == 42);
  }
  // it--
  {
    BoundedIter iter   = std::__make_static_packed_bounded_iter<Iter, decltype(array), std::size(array)>(Iter(b)) + 3;
    BoundedIter result = iter--;
    assert(*result == 43);
    assert(*iter == 42);
  }
  // it += n
  {
    BoundedIter iter    = std::__make_static_packed_bounded_iter<Iter, decltype(array), std::size(array)>(Iter(b));
    BoundedIter& result = (iter += 3);
    assert(&result == &iter);
    assert(*iter == 43);
  }
  // it + n
  {
    BoundedIter iter   = std::__make_static_packed_bounded_iter<Iter, decltype(array), std::size(array)>(Iter(b));
    BoundedIter result = iter + 3;
    assert(*iter == 40);
    assert(*result == 43);
  }
  // n + it
  {
    BoundedIter iter   = std::__make_static_packed_bounded_iter<Iter, decltype(array), std::size(array)>(Iter(b));
    BoundedIter result = 3 + iter;
    assert(*iter == 40);
    assert(*result == 43);
  }
  // it -= n
  {
    BoundedIter iter    = std::__make_static_packed_bounded_iter<Iter, decltype(array), std::size(array)>(Iter(b)) + 3;
    BoundedIter& result = (iter -= 3);
    assert(&result == &iter);
    assert(*iter == 40);
  }
  // it - n
  {
    BoundedIter iter   = std::__make_static_packed_bounded_iter<Iter, decltype(array), std::size(array)>(Iter(b)) + 3;
    BoundedIter result = iter - 3;
    assert(*iter == 43);
    assert(*result == 40);
  }
  // it - it
  {
    BoundedIter iter1     = std::__make_static_packed_bounded_iter<Iter, decltype(array), std::size(array)>(Iter(b));
    BoundedIter iter2     = std::__make_static_packed_bounded_iter<Iter, decltype(array), std::size(array)>(Iter(e));
    std::ptrdiff_t result = iter2 - iter1;
    assert(result == 5);
  }

  return true;
}

int main(int, char**) {
  tests<Foo*>();
  static_assert(tests<Foo*>());

  return 0;
}
