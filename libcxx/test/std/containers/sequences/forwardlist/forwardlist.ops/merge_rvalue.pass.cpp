//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <forward_list>

// void merge(forward_list&& x); // constexpr since C++26

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

  friend TEST_CONSTEXPR bool operator<(const value& lhs, const value& rhs) { return lhs.a < rhs.a; }
  friend TEST_CONSTEXPR bool operator==(const value& lhs, const value& rhs) { return lhs.a == rhs.a && lhs.b == rhs.b; }
};

TEST_CONSTEXPR_CXX26 bool test() {
  { // Basic merge operation.
    typedef int T;
    typedef std::forward_list<T> C;
    const T t1[] = {3, 5, 6, 7, 12, 13};
    const T t2[] = {0, 1, 2, 4, 8, 9, 10, 11, 14, 15};
    const T t3[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

    C c1(std::begin(t1), std::end(t1));
    C c2(std::begin(t2), std::end(t2));
    c1.merge(std::move(c2));
    assert(c2.empty());

    C c3(std::begin(t3), std::end(t3));
    assert(c1 == c3);
  }
  { // Pointers, references, and iterators should remain valid after merging.
    typedef int T;
    typedef std::forward_list<T> C;
    typedef T* P;
    typedef typename C::iterator I;
    const T to[3] = {0, 1, 2};

    C c2(std::begin(to), std::end(to));
    I io[3]                         = {c2.begin(), ++c2.begin(), ++ ++c2.begin()};
    std::reference_wrapper<T> ro[3] = {*io[0], *io[1], *io[2]};
    P po[3]                         = {&*io[0], &*io[1], &*io[2]};

    C c1;
    c1.merge(std::move(c2));
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
    const T t1[] = {{0, 0}, {2, 0}, {3, 0}};
    const T t2[] = {{0, 1}, {1, 1}, {2, 1}, {4, 1}};
    const T t3[] = {{0, 0}, {0, 1}, {1, 1}, {2, 0}, {2, 1}, {3, 0}, {4, 1}};

    C c1(std::begin(t1), std::end(t1));
    C c2(std::begin(t2), std::end(t2));
    c1.merge(std::move(c2));
    assert(c2.empty());

    C c3(std::begin(t3), std::end(t3));
    assert(c1 == c3);
  }
  { // Test with a different allocator.
    typedef int T;
    typedef std::forward_list<T, min_allocator<T>> C;
    const T t1[] = {3, 5, 6, 7, 12, 13};
    const T t2[] = {0, 1, 2, 4, 8, 9, 10, 11, 14, 15};
    const T t3[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

    C c1(std::begin(t1), std::end(t1));
    C c2(std::begin(t2), std::end(t2));
    c1.merge(std::move(c2));
    assert(c2.empty());

    C c3(std::begin(t3), std::end(t3));
    assert(c1 == c3);
  }

  { // LWG3088: Make sure self-merging does nothing.
    int a[] = {1, 2, 3, 4, 5};
    std::forward_list<int> c(std::begin(a), std::end(a));
    c.merge(std::move(c));
    assert(c == std::forward_list<int>(std::begin(a), std::end(a)));
  }

  return true;
}

int main(int, char**) {
  assert(test());
#if TEST_STD_VER >= 26
  static_assert(test());
#endif

  return 0;
}
