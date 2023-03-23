//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <forward_list>

// template <class Compare> void merge(forward_list&& x, Compare comp);

#include <forward_list>
#include <functional>
#include <iterator>
#include <vector>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

/// Helper for testing a stable sort.
///
/// The relation operator uses \ref a.
/// The equality operator uses \ref a and \ref b.
struct value {
  int a;
  int b;

  friend bool operator>(const value& lhs, const value& rhs) { return lhs.a > rhs.a; }
  friend bool operator==(const value& lhs, const value& rhs) { return lhs.a == rhs.a && lhs.b == rhs.b; }
};

int main(int, char**) {
  { // Basic merge operation.
    typedef int T;
    typedef std::forward_list<T> C;
    const T t1[] = {13, 12, 7, 6, 5, 3};
    const T t2[] = {15, 14, 11, 10, 9, 8, 4, 2, 1, 0};
    const T t3[] = {15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};

    C c1(std::begin(t1), std::end(t1));
    C c2(std::begin(t2), std::end(t2));
    c1.merge(std::move(c2), std::greater<T>());
    assert(c2.empty());

    C c3(std::begin(t3), std::end(t3));
    assert(c1 == c3);
  }
  { // Pointers, references, and iterators should remain valid after merging.
    typedef int T;
    typedef std::forward_list<T> C;
    typedef T* P;
    typedef typename C::iterator I;
    const T to[3] = {2, 1, 0};

    C c2(std::begin(to), std::end(to));
    I io[3] = {c2.begin(), ++c2.begin(), ++ ++c2.begin()};
    std::reference_wrapper<T> ro[3] = {*io[0], *io[1], *io[2]};
    P po[3] = {&*io[0], &*io[1], &*io[2]};

    C c1;
    c1.merge(std::move(c2), std::greater<T>());
    assert(c2.empty());

    for (std::size_t i = 0; i < 3; ++i) {
      assert(to[i] == *io[i]);
      assert(to[i] == ro[i].get());
      assert(to[i] == *po[i]);
    }
  }
  { // Sorting is stable.
    typedef value T;
    typedef std::forward_list<T> C;
    const T t1[] = {{3, 0}, {2, 0}, {0, 0}};
    const T t2[] = {{4, 1}, {2, 1}, {1, 1}, {0, 1}};
    const T t3[] = {{4, 1}, {3, 0}, {2, 0}, {2, 1}, {1, 1}, {0, 0}, {0, 1}};

    C c1(std::begin(t1), std::end(t1));
    C c2(std::begin(t2), std::end(t2));
    c1.merge(std::move(c2), std::greater<T>());
    assert(c2.empty());

    C c3(std::begin(t3), std::end(t3));
    assert(c1 == c3);
  }

  { // Test with a different allocator.
    typedef int T;
    typedef std::forward_list<T, min_allocator<T>> C;
    const T t1[] = {13, 12, 7, 6, 5, 3};
    const T t2[] = {15, 14, 11, 10, 9, 8, 4, 2, 1, 0};
    const T t3[] = {15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};

    C c1(std::begin(t1), std::end(t1));
    C c2(std::begin(t2), std::end(t2));
    c1.merge(std::move(c2), std::greater<T>());
    assert(c2.empty());

    C c3(std::begin(t3), std::end(t3));
    assert(c1 == c3);
  }

  return 0;
}
