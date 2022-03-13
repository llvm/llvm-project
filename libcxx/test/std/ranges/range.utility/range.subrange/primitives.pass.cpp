//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// class std::ranges::subrange;

#include <ranges>

#include <cassert>
#include "test_iterators.h"
#include "types.h"

constexpr bool test() {
  int buff[] = {1, 2, 3, 4, 5};

  {
    std::ranges::subrange<MoveOnlyForwardIter, int*> a(MoveOnlyForwardIter(buff), buff + 5, 5);
    assert(base(a.begin()) == buff);
    assert(!a.empty());
    assert(a.size() == 5);
  }

  {
    std::ranges::subrange<ForwardIter> b(ForwardIter(nullptr), ForwardIter(nullptr));
    assert(b.empty());
  }

  {
    std::ranges::subrange<ForwardIter> c{ForwardIter(buff), ForwardIter(buff)};
    assert(c.empty());
  }

  {
    std::ranges::subrange<ForwardIter> d(ForwardIter(buff), ForwardIter(buff + 1));
    assert(!d.empty());
  }

  {
    bool minusWasCalled = false;
    SizedSentinelForwardIter beg(buff, &minusWasCalled), end(buff + 5, &minusWasCalled);
    std::ranges::subrange<SizedSentinelForwardIter> e(beg, end, 5);
    assert(!e.empty());

    // Make sure that operator- is used to calculate size when possible.
    minusWasCalled = false;
    assert(e.size() == 5);
    assert(minusWasCalled);
  }

  return true;
}


int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
