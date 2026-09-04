//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// allocator:
// template <class... Args> void construct(pointer p, Args&&... args);

// In C++20, parts of std::allocator<T> have been removed.
// In C++17, they were deprecated.
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS
// REQUIRES: c++03 || c++11 || c++14 || c++17

#include <memory>
#include <cassert>

#include "test_macros.h"
#include "count_new.h"

int A_constructed = 0;

struct A {
  int data;
  A() { ++A_constructed; }

  A(const A&) { ++A_constructed; }

  explicit A(int) { ++A_constructed; }
  A(int, int*) { ++A_constructed; }

  ~A() { --A_constructed; }
};

int move_only_constructed = 0;

int main(int, char**) {
  globalMemCounter.reset();
  {
    std::allocator<A> a;
    assert(globalMemCounter.checkOutstandingNewEq(0));
    assert(A_constructed == 0);

    globalMemCounter.last_new_size = 0;
    A* ap                          = a.allocate(3);
    DoNotOptimize(ap);
    assert(globalMemCounter.checkOutstandingNewEq(1));
    assert(globalMemCounter.checkLastNewSizeEq(3 * sizeof(int)));
    assert(A_constructed == 0);

    a.construct(ap);
    assert(globalMemCounter.checkOutstandingNewEq(1));
    assert(A_constructed == 1);

    a.destroy(ap);
    assert(globalMemCounter.checkOutstandingNewEq(1));
    assert(A_constructed == 0);

    a.construct(ap, A());
    assert(globalMemCounter.checkOutstandingNewEq(1));
    assert(A_constructed == 1);

    a.destroy(ap);
    assert(globalMemCounter.checkOutstandingNewEq(1));
    assert(A_constructed == 0);

    a.construct(ap, 5);
    assert(globalMemCounter.checkOutstandingNewEq(1));
    assert(A_constructed == 1);

    a.destroy(ap);
    assert(globalMemCounter.checkOutstandingNewEq(1));
    assert(A_constructed == 0);

    a.construct(ap, 5, (int*)0);
    assert(globalMemCounter.checkOutstandingNewEq(1));
    assert(A_constructed == 1);

    a.destroy(ap);
    assert(globalMemCounter.checkOutstandingNewEq(1));
    assert(A_constructed == 0);

    a.deallocate(ap, 3);
    DoNotOptimize(ap);
    assert(globalMemCounter.checkOutstandingNewEq(0));
    assert(A_constructed == 0);
  }

  return 0;
}
