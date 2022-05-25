//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>
//
// constexpr iterator_type base() const; // Until C++20
// constexpr const Iterator& base() const & noexcept; // From C++20
// constexpr Iterator base() &&; // From C++20

#include <iterator>

#include <utility>
#include "test_iterators.h"
#include "test_macros.h"

struct MoveOnlyIterator {
  using It = int*;

  It it_;

  using iterator_category = std::input_iterator_tag;
  using value_type = int;
  using difference_type = std::ptrdiff_t;
  using reference = int&;

  TEST_CONSTEXPR explicit MoveOnlyIterator(It it) : it_(it) {}
  MoveOnlyIterator(MoveOnlyIterator&&) = default;
  MoveOnlyIterator& operator=(MoveOnlyIterator&&) = default;
  MoveOnlyIterator(const MoveOnlyIterator&) = delete;
  MoveOnlyIterator& operator=(const MoveOnlyIterator&) = delete;

  TEST_CONSTEXPR reference operator*() const { return *it_; }

  TEST_CONSTEXPR_CXX14 MoveOnlyIterator& operator++() { ++it_; return *this; }
  TEST_CONSTEXPR_CXX14 MoveOnlyIterator operator++(int) { return MoveOnlyIterator(it_++); }

  friend TEST_CONSTEXPR bool operator==(const MoveOnlyIterator& x, const MoveOnlyIterator& y) {return x.it_ == y.it_;}
  friend TEST_CONSTEXPR bool operator!=(const MoveOnlyIterator& x, const MoveOnlyIterator& y) {return x.it_ != y.it_;}

  friend TEST_CONSTEXPR It base(const MoveOnlyIterator& i) { return i.it_; }
};

#if TEST_STD_VER > 17
static_assert( std::input_iterator<MoveOnlyIterator>);
#endif
static_assert(!std::is_copy_constructible<MoveOnlyIterator>::value, "");

template <class It>
TEST_CONSTEXPR_CXX14 void test_one() {
  // Non-const lvalue.
  {
    int a[] = {1, 2, 3};

    auto i = std::move_iterator<It>(It(a));
#if TEST_STD_VER > 17
    ASSERT_SAME_TYPE(decltype(i.base()), const It&);
    ASSERT_NOEXCEPT(i.base());
#else
    ASSERT_SAME_TYPE(decltype(i.base()), It);
#endif
    assert(i.base() == It(a));

    ++i;
    assert(i.base() == It(a + 1));
  }

  // Const lvalue.
  {
    int a[] = {1, 2, 3};

    const auto i = std::move_iterator<It>(It(a));
#if TEST_STD_VER > 17
    ASSERT_SAME_TYPE(decltype(i.base()), const It&);
    ASSERT_NOEXCEPT(i.base());
#else
    ASSERT_SAME_TYPE(decltype(i.base()), It);
#endif
    assert(i.base() == It(a));
  }

  // Rvalue.
  {
    int a[] = {1, 2, 3};

    auto i = std::move_iterator<It>(It(a));
    ASSERT_SAME_TYPE(decltype(std::move(i).base()), It);
    assert(std::move(i).base() == It(a));
  }
}

TEST_CONSTEXPR_CXX14 bool test() {
  test_one<cpp17_input_iterator<int*> >();
  test_one<forward_iterator<int*> >();
  test_one<bidirectional_iterator<int*> >();
  test_one<random_access_iterator<int*> >();
  test_one<int*>();
  test_one<const int*>();
#if TEST_STD_VER > 17
  test_one<contiguous_iterator<int*>>();
#endif

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER > 14
  static_assert(test());
#endif

  return 0;
}
