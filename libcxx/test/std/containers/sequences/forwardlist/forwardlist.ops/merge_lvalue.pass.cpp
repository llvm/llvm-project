//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <forward_list>

// void merge(forward_list& x);

#include <forward_list>
#include <iterator>
#include <vector>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

#if TEST_STD_VER >= 11
#  include <functional>
#endif

/// Helper struct to test a stable sort.
///
/// relation comparison uses a.
/// equality comparison uses a and b.
struct value {
  int a;
  int b;

  friend bool operator<(const value& lhs, const value& rhs) { return lhs.a < rhs.a; }
  friend bool operator==(const value& lhs, const value& rhs) { return lhs.a == rhs.a && lhs.b == rhs.b; }
};

int main(int, char**) {
  { // Basic merge operation.
    typedef int T;
    typedef std::forward_list<T> C;
    const T t1[] = {3, 5, 6, 7, 12, 13};
    const T t2[] = {0, 1, 2, 4, 8, 9, 10, 11, 14, 15};
    const T t3[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

    C c1(std::begin(t1), std::end(t1));
    C c2(std::begin(t2), std::end(t2));
    c1.merge(c2);
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
    I io[3] = {c2.begin(), ++c2.begin(), ++ ++c2.begin()};
#if TEST_STD_VER >= 11
    std::reference_wrapper<T> ro[3] = {*io[0], *io[1], *io[2]};
#endif
    P po[3] = {&*io[0], &*io[1], &*io[2]};

    C c1;
    c1.merge(c2);
    assert(c2.empty());

    for (std::size_t i = 0; i < 3; ++i) {
      assert(to[i] == *io[i]);
#if TEST_STD_VER >= 11
      assert(to[i] == ro[i].get());
#endif
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
    c1.merge(c2);
    assert(c2.empty());

    C c3(std::begin(t3), std::end(t3));
    assert(c1 == c3);
  }
#if TEST_STD_VER >= 11
  { // Test with a different allocator.
    typedef int T;
    typedef std::forward_list<T, min_allocator<T>> C;
    const T t1[] = {3, 5, 6, 7, 12, 13};
    const T t2[] = {0, 1, 2, 4, 8, 9, 10, 11, 14, 15};
    const T t3[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

    C c1(std::begin(t1), std::end(t1));
    C c2(std::begin(t2), std::end(t2));
    c1.merge(c2);
    assert(c2.empty());

    C c3(std::begin(t3), std::end(t3));
    assert(c1 == c3);
  }
#endif

  return 0;
}
