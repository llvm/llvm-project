//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20

#include <__algorithm/iterator_operations.h>
#include <cassert>
#include <cstddef>

#include "test_macros.h"
#include "test_range.h"

template <class I, class S, bool Sized = false>
struct test_subrange {
  I first;
  S last;
  std::size_t n = 0;

  bool flag = false;

  I begin() { return first; }
  S end() { return last; }

  std::size_t size()
    requires Sized
  {
    flag = true;
    return n;
  }
};

template <class I, class S>
test_subrange(I, S) -> test_subrange<I, S>;

template <class I, class S>
test_subrange(I, S, std::size_t) -> test_subrange<I, S, true>;

constexpr int data[] = {1, 2, 3};

using Iter = random_access_iterator<const int*>;
using Sent = sentinel_wrapper<Iter>;

int main(int, char**) {
  test_subrange r0{Iter(data), Iter(data + 3), 3};
  std::_IterOps<std::_RangeAlgPolicy>::__end(r0);
  assert(r0.flag == false);

  test_subrange r1{Iter(data), Sent(Iter(data + 3)), 3};
  std::_IterOps<std::_RangeAlgPolicy>::__end(r1);
  assert(r1.flag == true);

  test_subrange r2{Iter(data), Sent(Iter(data + 3))};
  std::_IterOps<std::_RangeAlgPolicy>::__end(r2);
  assert(r2.flag == false);
}
