//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// move_iterator

// template <InputIterator Iter1, InputIterator Iter2>
//   requires HasEqualTo<Iter1, Iter2>
//   bool
//   operator!=(const move_iterator<Iter1>& x, const move_iterator<Iter2>& y);
//
//  constexpr in C++17

#include <iterator>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"

// In C++17, move_iterator's operator!= calls the underlying iterator's operator!=
// In C++20, move_iterator's operator== calls the underlying iterator's operator==
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
#if TEST_STD_VER > 17
  friend constexpr bool operator==(const CustomIt& a, const CustomIt& b) { return a.p_ == b.p_; }
  friend bool operator!=(const CustomIt& a, const CustomIt& b) = delete;
#else
  friend TEST_CONSTEXPR_CXX17 bool operator!=(const CustomIt& a, const CustomIt& b) { return a.p_ != b.p_; }
#endif
  int *p_ = nullptr;
};

template <class It>
TEST_CONSTEXPR_CXX17 void test_one()
{
  int a[] = {3, 1, 4};
  const std::move_iterator<It> r1 = std::move_iterator<It>(It(a));
  const std::move_iterator<It> r2 = std::move_iterator<It>(It(a));
  const std::move_iterator<It> r3 = std::move_iterator<It>(It(a + 2));
  ASSERT_SAME_TYPE(decltype(r1 != r2), bool);
  assert(!(r1 != r1));
  assert(!(r1 != r2));
  assert(!(r2 != r1));
  assert( (r1 != r3));
  assert( (r3 != r1));
}

TEST_CONSTEXPR_CXX17 bool test()
{
  test_one<CustomIt>();
  test_one<cpp17_input_iterator<int*> >();
  test_one<forward_iterator<int*> >();
  test_one<bidirectional_iterator<int*> >();
  test_one<random_access_iterator<int*> >();
  test_one<int*>();
  test_one<const int*>();

#if TEST_STD_VER > 17
  test_one<contiguous_iterator<int*>>();
  test_one<three_way_contiguous_iterator<int*>>();
#endif

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER > 14
  static_assert(test());
#endif

  return 0;
}
