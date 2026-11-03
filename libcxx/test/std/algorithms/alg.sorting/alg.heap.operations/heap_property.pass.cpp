//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// LWG2166: Heap property underspecified?
// [alg.heap.operations]/1: With N = b - a, for all i, 0 < i < N,
// bool(comp(a[floor((i - 1) / 2)], a[i])) is false.

#include <algorithm>
#include <cassert>
#include <iterator>

#include "test_macros.h"

// For all i >= 0,
// comp(a[i], a[L]) is false whenever L = 2*i+1 < b-a,
// and
// comp(a[i], a[R]) is false whenever R = 2*i+2 < b-a.
template <class Comp, class Iter>
TEST_CONSTEXPR_CXX20 bool lwg2166_heap(Iter a, Iter b, Comp cmp) {
  using difference_type = std::iterator_traits<Iter>::difference_type;
  const difference_type n = b - a;
  for (difference_type i = 1; i < n; ++i)
    if (cmp(a[(i - 1) / 2], a[i]))
      return false;
  return true;
}

TEST_CONSTEXPR_CXX20 bool test() {
  auto cmp = [](int x, int y) { return x < y; };

  // std::pop_heap must preserve the normative predicate.
  {
    int a[] = {42, 4, 5, 1, 2};
    int* const last = std::end(a);
    assert(lwg2166_heap(a, last, cmp));
    assert(std::is_heap_until(a, last, cmp) == last);

    for (int* end = last; end - a > 1; --end) {
      std::pop_heap(a, end, cmp);
      assert(lwg2166_heap(a, end - 1, cmp));
    }
  }

  // std::push_heap must preserve the normative predicate.
  {
    int a[] = {42, 4, 5, 1, 2};
    const std::ptrdiff_t num = std::end(a) - a;

    for (std::ptrdiff_t len = 1; len <= num; ++len) {
      std::push_heap(a, a + len, cmp);
      assert(lwg2166_heap(a, a + len, cmp));
    }
  } 

  // all equal elements.
  {
    int eq[] = {0, 0, 0, 0, 0};
    assert(lwg2166_heap(std::begin(eq), std::end(eq), cmp));
    assert(std::is_heap(std::begin(eq), std::end(eq), cmp));
  }

  // is_heap_until must report the first index that breaks the binary layout.
  {
    int a[] = {3, 2, 1, 42};
    assert(!lwg2166_heap(std::begin(a), std::end(a), cmp));
    assert(std::is_heap_until(std::begin(a), std::end(a), cmp) == a + 3);
  }

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 20
  static_assert(test());
#endif
}
