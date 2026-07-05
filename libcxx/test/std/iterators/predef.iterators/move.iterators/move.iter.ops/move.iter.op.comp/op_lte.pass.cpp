//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// move_iterator

// template <class Iter1, class Iter2>
//   bool operator<=(const move_iterator<Iter1>& x, const move_iterator<Iter2>& y);
//
//  constexpr in C++17

#include <iterator>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"

// move_iterator's operator<= calls the underlying iterator's operator<=
struct CustomIt {
  using value_type = int;
  using difference_type = int;
  using reference = int&;
  using pointer = int*;
  using iterator_category = std::input_iterator_tag;
  CustomIt() = default;
  TEST_CONSTEXPR_CXX17 explicit CustomIt(int* p) : p_(p) {}
  int& operator*() const;
  CustomIt& operator++();
  CustomIt operator++(int);
  TEST_CONSTEXPR_CXX17 friend bool operator<=(const CustomIt& a, const CustomIt& b) { return a.p_ <= b.p_; }
  int *p_ = nullptr;
};

template <class It>
TEST_CONSTEXPR_CXX17 void test_one()
{
  int a[] = {3, 1, 4};
  const std::move_iterator<It> r1 = std::move_iterator<It>(It(a));
  const std::move_iterator<It> r2 = std::move_iterator<It>(It(a + 2));
  ASSERT_SAME_TYPE(decltype(r1 <= r2), bool);
  assert( (r1 <= r1));
  assert( (r1 <= r2));
  assert(!(r2 <= r1));
}

TEST_CONSTEXPR_CXX17 bool test()
{
  test_one<CustomIt>();
  test_one<int*>();
  test_one<const int*>();
  test_one<random_access_iterator<int*> >();
#if TEST_STD_VER > 17
  test_one<contiguous_iterator<int*>>();
#endif
  return true;
}

int main(int, char**)
{
  assert(test());
#if TEST_STD_VER > 14
  static_assert(test());
#endif

  return 0;
}
