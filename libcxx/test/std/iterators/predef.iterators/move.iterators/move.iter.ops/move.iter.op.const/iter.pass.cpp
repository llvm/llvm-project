//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// move_iterator

// explicit move_iterator(Iter i);
//
//  constexpr in C++17

#include <iterator>
#include <cassert>
#include <utility>

#include "test_macros.h"
#include "test_iterators.h"

template <class It>
TEST_CONSTEXPR_CXX17 bool test()
{
  static_assert( std::is_constructible<std::move_iterator<It>, const It&>::value, "");
  static_assert( std::is_constructible<std::move_iterator<It>, It&&>::value, "");
  static_assert(!std::is_convertible<const It&, std::move_iterator<It> >::value, "");
  static_assert(!std::is_convertible<It&&, std::move_iterator<It> >::value, "");

  char s[] = "123";
  {
    It it = It(s);
    std::move_iterator<It> r(it);
    assert(base(r.base()) == s);
  }
  {
    It it = It(s);
    std::move_iterator<It> r(std::move(it));
    assert(base(r.base()) == s);
  }
  return true;
}

template <class It>
TEST_CONSTEXPR_CXX17 bool test_moveonly()
{
  static_assert(!std::is_constructible<std::move_iterator<It>, const It&>::value, "");
  static_assert( std::is_constructible<std::move_iterator<It>, It&&>::value, "");
  static_assert(!std::is_convertible<const It&, std::move_iterator<It> >::value, "");
  static_assert(!std::is_convertible<It&&, std::move_iterator<It> >::value, "");

  char s[] = "123";
  {
    It it = It(s);
    std::move_iterator<It> r(std::move(it));
    assert(base(r.base()) == s);
  }
  return true;
}

int main(int, char**)
{
  test<cpp17_input_iterator<char*> >();
  test<forward_iterator<char*> >();
  test<bidirectional_iterator<char*> >();
  test<random_access_iterator<char*> >();
  test<char*>();
  test<const char*>();

#if TEST_STD_VER > 14
  static_assert(test<cpp17_input_iterator<char*>>());
  static_assert(test<forward_iterator<char*>>());
  static_assert(test<bidirectional_iterator<char*>>());
  static_assert(test<random_access_iterator<char*>>());
  static_assert(test<char*>());
  static_assert(test<const char*>());
#endif

#if TEST_STD_VER > 17
  test<contiguous_iterator<char*>>();
  test_moveonly<cpp20_input_iterator<char*>>();
  static_assert(test<contiguous_iterator<char*>>());
  static_assert(test_moveonly<cpp20_input_iterator<char*>>());
#endif

  return 0;
}
