//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// move_iterator

// constexpr auto operator++(int); // Return type was move_iterator until C++20

#include <iterator>
#include <cassert>
#include <utility>

#include "test_macros.h"
#include "test_iterators.h"

#if TEST_STD_VER > 17
template <class It>
void test_single_pass(It i, It x) {
  std::move_iterator<It> r(std::move(i));
  r++;
  assert(std::move(r).base() == x);
}
#endif

template <class It>
void test(It i, It x) {
    std::move_iterator<It> r(i);
    std::move_iterator<It> rr = r++;
    assert(r.base() == x);
    assert(rr.base() == i);
}

int main(int, char**) {
  char s[] = "123";
#if TEST_STD_VER > 17
  test_single_pass(cpp17_input_iterator<char*>(s), cpp17_input_iterator<char*>(s + 1));
#else
  test(cpp17_input_iterator<char*>(s), cpp17_input_iterator<char*>(s+1));
#endif
  test(forward_iterator<char*>(s), forward_iterator<char*>(s+1));
  test(bidirectional_iterator<char*>(s), bidirectional_iterator<char*>(s+1));
  test(random_access_iterator<char*>(s), random_access_iterator<char*>(s+1));
  test(s, s+1);

#if TEST_STD_VER > 14
  {
    constexpr const char *p = "123456789";
    typedef std::move_iterator<const char *> MI;
    constexpr MI it1 = std::make_move_iterator(p);
    constexpr MI it2 = std::make_move_iterator(p+1);
    static_assert(it1 != it2, "");
    constexpr MI it3 = std::make_move_iterator(p) ++;
    static_assert(it1 == it3, "");
    static_assert(it2 != it3, "");
  }
#endif

#if TEST_STD_VER > 17
  // Forward iterators return a copy.
  {
    int a[] = {1, 2, 3};
    using MoveIter = std::move_iterator<forward_iterator<int*>>;

    MoveIter i = MoveIter(forward_iterator<int*>(a));
    ASSERT_SAME_TYPE(decltype(i++), MoveIter);
    auto j = i++;
    assert(base(j.base()) == a);
    assert(base(i.base()) == a + 1);
  }

  // Non-forward iterators return void.
  {
    int a[] = {1, 2, 3};
    using MoveIter = std::move_iterator<cpp20_input_iterator<int*>>;

    MoveIter i = MoveIter(cpp20_input_iterator<int*>(a));
    ASSERT_SAME_TYPE(decltype(i++), void);
    i++;
    assert(base(i.base()) == a + 1);
  }
#endif

  return 0;
}
