//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <vector>

// class vector<bool, Allocator>

// template<class T, class Allocator>
//   constexpr synth-three-way-result<T> operator<=>(const vector<T, Allocator>& x,
//                                                   const vector<T, Allocator>& y);

#include <cassert>
#include <vector>

#include "test_comparisons.h"

constexpr bool test_sequence_container_spaceship_vectorbool() {
  // Empty containers
  {
    std::vector<bool> l1;
    std::vector<bool> l2;
    assert(testOrder(l1, l2, std::strong_ordering::equivalent));
  }
  // Identical contents
  {
    std::vector<bool> t1{true, true};
    std::vector<bool> t2{true, true};
    assert(testOrder(t1, t2, std::strong_ordering::equivalent));

    std::vector<bool> f1{false, false};
    std::vector<bool> f2{false, false};
    assert(testOrder(f1, f2, std::strong_ordering::equivalent));
  }
  // Less, due to contained values
  {
    std::vector<bool> l1{true, false};
    std::vector<bool> l2{true, true};
    assert(testOrder(l1, l2, std::strong_ordering::less));
  }
  // Greater, due to contained values
  {
    std::vector<bool> l1{true, true};
    std::vector<bool> l2{true, false};
    assert(testOrder(l1, l2, std::strong_ordering::greater));
  }
  // Shorter list
  {
    std::vector<bool> l1{true};
    std::vector<bool> l2{true, false};
    assert(testOrder(l1, l2, std::strong_ordering::less));

    std::vector<bool> t1{true};
    std::vector<bool> t2{true, true};
    assert(testOrder(t1, t2, std::strong_ordering::less));

    std::vector<bool> f1{false};
    std::vector<bool> f2{false, false};
    assert(testOrder(f1, f2, std::strong_ordering::less));

    std::vector<bool> e;
    assert(testOrder(e, t1, std::strong_ordering::less));
    assert(testOrder(e, f1, std::strong_ordering::less));
  }
  // Longer list
  {
    std::vector<bool> l1{true, false};
    std::vector<bool> l2{true};
    assert(testOrder(l1, l2, std::strong_ordering::greater));

    std::vector<bool> t1{true, true};
    std::vector<bool> t2{true};
    assert(testOrder(t1, t2, std::strong_ordering::greater));

    std::vector<bool> f1{false, false};
    std::vector<bool> f2{false};
    assert(testOrder(f1, f2, std::strong_ordering::greater));

    std::vector<bool> e;
    assert(testOrder(t2, e, std::strong_ordering::greater));
    assert(testOrder(f2, e, std::strong_ordering::greater));
  }

  return true;
}

int main(int, char**) {
  assert(test_sequence_container_spaceship_vectorbool());
  static_assert(test_sequence_container_spaceship_vectorbool());
  return 0;
}
