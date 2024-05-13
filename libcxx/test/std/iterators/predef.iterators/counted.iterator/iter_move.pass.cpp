//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// friend constexpr iter_rvalue_reference_t<I>
//   iter_move(const counted_iterator& i)
//     noexcept(noexcept(ranges::iter_move(i.current)))
//     requires input_iterator<I>;

#include <iterator>

#include "test_macros.h"
#include "test_iterators.h"

template<bool IsNoexcept>
class HasNoexceptIterMove
{
  int *it_;

public:
  typedef          std::input_iterator_tag                      iterator_category;
  typedef int                                                   value_type;
  typedef typename std::iterator_traits<int *>::difference_type difference_type;
  typedef int *                                                 pointer;
  typedef int &                                                 reference;

  constexpr int *base() const {return it_;}

  HasNoexceptIterMove() = default;
  explicit constexpr HasNoexceptIterMove(int *it) : it_(it) {}

  constexpr reference operator*() const noexcept(IsNoexcept) { return *it_; }

  constexpr HasNoexceptIterMove& operator++() {++it_; return *this;}
  constexpr HasNoexceptIterMove operator++(int)
      {HasNoexceptIterMove tmp(*this); ++(*this); return tmp;}
};


constexpr bool test() {
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    auto iter1 = cpp17_input_iterator<int*>(buffer);
    auto commonIter1 = std::counted_iterator<decltype(iter1)>(iter1, 8);
    assert(std::ranges::iter_move(commonIter1) == 1);
    ASSERT_SAME_TYPE(decltype(std::ranges::iter_move(commonIter1)), int&&);
  }
  {
    auto iter1 = forward_iterator<int*>(buffer);
    auto commonIter1 = std::counted_iterator<decltype(iter1)>(iter1, 8);
    assert(std::ranges::iter_move(commonIter1) == 1);
    ASSERT_SAME_TYPE(decltype(std::ranges::iter_move(commonIter1)), int&&);
  }
  {
    auto iter1 = random_access_iterator<int*>(buffer);
    auto commonIter1 = std::counted_iterator<decltype(iter1)>(iter1, 8);
    assert(std::ranges::iter_move(commonIter1) == 1);
    ASSERT_SAME_TYPE(decltype(std::ranges::iter_move(commonIter1)), int&&);
  }

  // Test noexceptness.
  {
    static_assert( noexcept(std::ranges::iter_move(std::declval<std::counted_iterator<HasNoexceptIterMove<true>>>())));
    static_assert(!noexcept(std::ranges::iter_move(std::declval<std::counted_iterator<HasNoexceptIterMove<false>>>())));
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
