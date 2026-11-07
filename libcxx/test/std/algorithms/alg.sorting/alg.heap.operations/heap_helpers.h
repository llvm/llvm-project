//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

#include <cassert>
#include <iterator>

#include "test_macros.h"

// LWG2166: Heap property underspecified?
// [alg.heap.operations]/1: With N = b - a, for all i, 0 < i < N,
// bool(comp(a[floor((i - 1) / 2)], a[i])) is false.
template <class Comp, class Iter>
TEST_CONSTEXPR_CXX20 bool satisfies_heap_property(Iter a, Iter b, Comp cmp) {
  typedef typename std::iterator_traits<Iter>::difference_type difference_type;
  const difference_type n = b - a;
  for (difference_type i = 1; i < n; ++i)
    if (cmp(a[(i - 1) / 2], a[i]))
      return false;
  return true;
}

template <class Iter>
TEST_CONSTEXPR_CXX20 bool satisfies_heap_property(Iter a, Iter b) {
    typedef typename std::iterator_traits<Iter>::value_type value_type;
    return satisfies_heap_property(a, b, std::less<value_type>());
}
