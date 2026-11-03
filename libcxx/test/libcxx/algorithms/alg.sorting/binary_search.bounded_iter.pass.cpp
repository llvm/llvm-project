//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// The binary-search algorithms unwrap libc++'s bounded iterators after validating the input range once. Check that
// - only __bounded_iter and __static_bounded_iter are unwrapped,
// - the results are correct and keep the bounds of the input iterators (including in constant expressions), and
// - other users of iterator unwrapping are unaffected.

#include <__algorithm/unwrap_iter.h>
#include <__iterator/bounded_iter.h>
#include <__iterator/static_bounded_iter.h>
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>

#include "test_iterators.h"
#include "test_macros.h"

template <class Iter>
using UnwrapBoundedT = decltype(std::__unwrap_bounded_iter(std::declval<Iter>(), std::ptrdiff_t()));

// Only libc++'s own bounded iterators are unwrapped.
static_assert(std::is_same<UnwrapBoundedT<std::__bounded_iter<int*> >, int*>::value, "");
static_assert(std::is_same<UnwrapBoundedT<std::__bounded_iter<const int*> >, const int*>::value, "");
static_assert(std::is_same<UnwrapBoundedT<std::__static_bounded_iter<int*, 4> >, int*>::value, "");
static_assert(std::is_same<UnwrapBoundedT<int*>, int*>::value, "");
static_assert(std::is_same<UnwrapBoundedT<std::__wrap_iter<int*> >, std::__wrap_iter<int*> >::value, "");
static_assert(std::is_same<UnwrapBoundedT<std::reverse_iterator<std::reverse_iterator<int*> > >,
                           std::reverse_iterator<std::reverse_iterator<int*> > >::value,
              "");
static_assert(std::is_same<UnwrapBoundedT<forward_iterator<int*> >, forward_iterator<int*> >::value, "");
static_assert(std::is_same<UnwrapBoundedT<random_access_iterator<int*> >, random_access_iterator<int*> >::value, "");

template <class Iter>
TEST_CONSTEXPR_CXX20 void check_searches(Iter first, Iter last) {
  // [first, last) holds {1, 2, 2, 2, 3, 5, 8}.
  assert(std::lower_bound(first, last, 2) == first + 1);
  assert(std::upper_bound(first, last, 2) == first + 4);
  assert(std::lower_bound(first, last, 0) == first);
  assert(std::upper_bound(first, last, 9) == last);
  assert(std::equal_range(first, last, 2) == std::make_pair(first + 1, first + 4));
  assert(std::equal_range(first, last, 4) == std::make_pair(first + 5, first + 5));
  assert(std::binary_search(first, last, 5));
  assert(!std::binary_search(first, last, 4));
#if TEST_STD_VER >= 20
  assert(std::ranges::lower_bound(first, last, 3) == first + 4);
  assert(std::ranges::upper_bound(first, last, 3) == first + 5);
  auto range = std::ranges::equal_range(first, last, 2);
  assert(range.begin() == first + 1 && range.end() == first + 4);
  assert(std::ranges::binary_search(first, last, 8));
#endif

  // The results are usable like any other iterator into the range: they can be dereferenced and moved within the
  // original bounds.
  Iter it = std::lower_bound(first, last, 3);
  assert(*it == 3);
  assert(*(it - 4) == 1);
  assert(it + 3 == last);

  // Empty ranges.
  assert(std::lower_bound(first, first, 2) == first);
  assert(std::equal_range(last, last, 2) == std::make_pair(last, last));
}

TEST_CONSTEXPR_CXX20 bool test() {
  int array[] = {1, 2, 2, 2, 3, 5, 8};
  int* b      = array;
  int* e      = array + 7;

  check_searches(std::__make_bounded_iter(b, b, e), std::__make_bounded_iter(e, b, e));
  check_searches(std::__make_static_bounded_iter<7>(b, b), std::__make_static_bounded_iter<7>(e, b));

  // A subrange of a larger bounded range.
  int bigger[] = {0, 1, 2, 2, 2, 3, 5, 8, 9};
  check_searches(std::__make_bounded_iter(bigger + 1, bigger, bigger + 9),
                 std::__make_bounded_iter(bigger + 8, bigger, bigger + 9));

  // Non-bounded iterators still work.
  check_searches(b, e);
  check_searches(random_access_iterator<int*>(b), random_access_iterator<int*>(e));
  return true;
}

// An output iterator whose difference_type is void, as with std::iterator<std::output_iterator_tag, void, void, void,
// void>. std::copy unwraps its output iterator, which must keep working.
struct CountingOutputIterator {
  using iterator_category = std::output_iterator_tag;
  using value_type        = void;
  using difference_type   = void;
  using pointer           = void;
  using reference         = void;

  int* count;
  CountingOutputIterator& operator*() { return *this; }
  CountingOutputIterator& operator=(int) {
    ++*count;
    return *this;
  }
  CountingOutputIterator& operator++() { return *this; }
  CountingOutputIterator operator++(int) { return *this; }
};

void test_other_unwrapping_users() {
  int array[] = {1, 2, 2, 3};

  int count = 0;
  std::copy(array, array + 4, CountingOutputIterator{&count});
  assert(count == 4);

  // reverse_iterator<reverse_iterator<It>> has its own __unwrap_iter_impl specialization.
  typedef std::reverse_iterator<std::reverse_iterator<int*> > DoublyReversed;
  DoublyReversed first((std::reverse_iterator<int*>(array)));
  DoublyReversed last((std::reverse_iterator<int*>(array + 4)));
  assert(std::lower_bound(first, last, 2) - first == 1);
  assert(std::upper_bound(first, last, 2) - first == 3);
  assert(std::equal_range(first, last, 2) == std::make_pair(first + 1, first + 3));
  assert(std::binary_search(first, last, 3));
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 20
  static_assert(test());
#endif
  test_other_unwrapping_users();
  return 0;
}
