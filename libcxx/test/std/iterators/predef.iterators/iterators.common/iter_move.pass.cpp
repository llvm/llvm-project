//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// friend iter_rvalue_reference_t<I> iter_move(const common_iterator& i)
//   noexcept(noexcept(ranges::iter_move(declval<const I&>())))
//     requires input_iterator<I>;

#include <iterator>
#include <cassert>
#include <type_traits>

#include "test_iterators.h"
#include "test_macros.h"

struct IterMovingIt {
  using value_type = int;
  using difference_type = int;
  explicit IterMovingIt() = default;
  IterMovingIt(const IterMovingIt&); // copyable, but this test shouldn't make copies
  IterMovingIt(IterMovingIt&&) = default;
  IterMovingIt& operator=(const IterMovingIt&);
  int& operator*() const;
  constexpr IterMovingIt& operator++() { return *this; }
  IterMovingIt operator++(int);
  friend constexpr int iter_move(const IterMovingIt&) {
    return 42;
  }
  bool operator==(std::default_sentinel_t) const;
};
static_assert(std::input_iterator<IterMovingIt>);

constexpr bool test() {
  {
    using It = int*;
    using CommonIt = std::common_iterator<It, sentinel_wrapper<It>>;
    int a[] = {1, 2, 3};
    CommonIt it = CommonIt(It(a));
    ASSERT_NOEXCEPT(iter_move(it));
    ASSERT_NOEXCEPT(std::ranges::iter_move(it));
    ASSERT_SAME_TYPE(decltype(iter_move(it)), int&&);
    ASSERT_SAME_TYPE(decltype(std::ranges::iter_move(it)), int&&);
    assert(iter_move(it) == 1);
    if (!std::is_constant_evaluated()) {
      ++it;
      assert(iter_move(it) == 2);
    }
  }
  {
    using It = const int*;
    using CommonIt = std::common_iterator<It, sentinel_wrapper<It>>;
    int a[] = {1, 2, 3};
    CommonIt it = CommonIt(It(a));
    ASSERT_NOEXCEPT(iter_move(it));
    ASSERT_NOEXCEPT(std::ranges::iter_move(it));
    ASSERT_SAME_TYPE(decltype(iter_move(it)), const int&&);
    ASSERT_SAME_TYPE(decltype(std::ranges::iter_move(it)), const int&&);
    assert(iter_move(it) == 1);
    if (!std::is_constant_evaluated()) {
      ++it;
      assert(iter_move(it) == 2);
    }
  }
  {
    using It = IterMovingIt;
    using CommonIt = std::common_iterator<It, std::default_sentinel_t>;
    CommonIt it = CommonIt(It());
    ASSERT_NOT_NOEXCEPT(iter_move(it));
    ASSERT_NOT_NOEXCEPT(std::ranges::iter_move(it));
    ASSERT_SAME_TYPE(decltype(iter_move(it)), int);
    ASSERT_SAME_TYPE(decltype(std::ranges::iter_move(it)), int);
    assert(iter_move(it) == 42);
    if (!std::is_constant_evaluated()) {
      ++it;
      assert(iter_move(it) == 42);
    }
  }
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
