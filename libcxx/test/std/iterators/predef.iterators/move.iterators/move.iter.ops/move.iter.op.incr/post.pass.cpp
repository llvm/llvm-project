//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// move_iterator

// move_iterator operator++(int);
//
//  constexpr in C++17

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
void
test(It i, It x)
{
    std::move_iterator<It> r(i);
    std::move_iterator<It> rr = r++;
    assert(r.base() == x);
    assert(rr.base() == i);
}

int main(int, char**)
{
    char s[] = "123";
#if TEST_STD_VER > 17
    test_single_pass(input_iterator<char*>(s), input_iterator<char*>(s + 1));
#else
    test(input_iterator<char*>(s), input_iterator<char*>(s+1));
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

  return 0;
}
