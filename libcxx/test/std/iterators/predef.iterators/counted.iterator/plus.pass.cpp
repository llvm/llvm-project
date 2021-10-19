//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts

// constexpr counted_iterator operator+(iter_difference_t<I> n) const
//     requires random_access_iterator<I>;
// friend constexpr counted_iterator operator+(
//   iter_difference_t<I> n, const counted_iterator& x)
//     requires random_access_iterator<I>;
// constexpr counted_iterator& operator+=(iter_difference_t<I> n)
//     requires random_access_iterator<I>;

#include <iterator>

#include "test_macros.h"
#include "test_iterators.h"

template<class Iter>
concept PlusEnabled = requires(Iter& iter) {
  iter + 1;
};

template<class Iter>
concept PlusEqEnabled = requires(Iter& iter) {
  iter += 1;
};

constexpr bool test() {
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    {
      using Counted = std::counted_iterator<random_access_iterator<int*>>;
      std::counted_iterator iter(random_access_iterator<int*>{buffer}, 8);
      assert(iter + 2 == Counted(random_access_iterator<int*>{buffer + 2}, 6));
      assert(iter + 0 == Counted(random_access_iterator<int*>{buffer}, 8));

      ASSERT_SAME_TYPE(decltype(iter + 2), Counted);
    }
    {
      using Counted = const std::counted_iterator<random_access_iterator<int*>>;
      const std::counted_iterator iter(random_access_iterator<int*>{buffer}, 8);
      assert(iter + 8 == Counted(random_access_iterator<int*>{buffer + 8}, 0));
      assert(iter + 0 == Counted(random_access_iterator<int*>{buffer}, 8));

      ASSERT_SAME_TYPE(decltype(iter + 2), std::remove_const_t<Counted>);
    }
  }

  {
    {
      using Counted = std::counted_iterator<random_access_iterator<int*>>;
      std::counted_iterator iter(random_access_iterator<int*>{buffer}, 8);
      assert(2 + iter == Counted(random_access_iterator<int*>{buffer + 2}, 6));
      assert(0 + iter == Counted(random_access_iterator<int*>{buffer}, 8));

      ASSERT_SAME_TYPE(decltype(iter + 2), Counted);
    }
    {
      using Counted = const std::counted_iterator<random_access_iterator<int*>>;
      const std::counted_iterator iter(random_access_iterator<int*>{buffer}, 8);
      assert(8 + iter == Counted(random_access_iterator<int*>{buffer + 8}, 0));
      assert(0 + iter == Counted(random_access_iterator<int*>{buffer}, 8));

      ASSERT_SAME_TYPE(decltype(iter + 2), std::remove_const_t<Counted>);
    }
  }

  {
    {
      using Counted = std::counted_iterator<random_access_iterator<int*>>;
      std::counted_iterator iter(random_access_iterator<int*>{buffer}, 8);
      assert((iter += 2) == Counted(random_access_iterator<int*>{buffer + 2}, 6));
      assert((iter += 0) == Counted(random_access_iterator<int*>{buffer + 2}, 6));

      ASSERT_SAME_TYPE(decltype(iter += 2), Counted&);
    }
    {
      using Counted = std::counted_iterator<contiguous_iterator<int*>>;
      std::counted_iterator iter(contiguous_iterator<int*>{buffer}, 8);
      assert((iter += 8) == Counted(contiguous_iterator<int*>{buffer + 8}, 0));
      assert((iter += 0) == Counted(contiguous_iterator<int*>{buffer + 8}, 0));

      ASSERT_SAME_TYPE(decltype(iter += 2), Counted&);
    }
    {
      static_assert( PlusEnabled<std::counted_iterator<random_access_iterator<int*>>>);
      static_assert(!PlusEnabled<std::counted_iterator<bidirectional_iterator<int*>>>);

      static_assert( PlusEqEnabled<      std::counted_iterator<random_access_iterator<int*>>>);
      static_assert(!PlusEqEnabled<const std::counted_iterator<random_access_iterator<int*>>>);
    }
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
