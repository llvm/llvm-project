//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// move_iterator

#include <iterator>

#include "test_iterators.h"
#include "test_macros.h"

void test()
{
  {
    using iterator = std::move_iterator<cpp17_input_iterator<int*>>;

    LIBCPP_STATIC_ASSERT(!std::default_initializable<iterator>);
    static_assert( std::copyable<iterator>);
    static_assert( std::input_iterator<iterator>);
    static_assert(!std::forward_iterator<iterator>);
    static_assert(!std::sentinel_for<iterator, iterator>); // not copyable
    static_assert(!std::sized_sentinel_for<iterator, iterator>);
    static_assert(!std::indirectly_movable<int*, iterator>);
    static_assert(!std::indirectly_movable_storable<int*, iterator>);
    static_assert(!std::indirectly_copyable<int*, iterator>);
    static_assert(!std::indirectly_copyable_storable<int*, iterator>);
    static_assert( std::indirectly_readable<iterator>);
    static_assert(!std::indirectly_writable<iterator, int>);
    static_assert( std::indirectly_swappable<iterator, iterator>);
  }
  {
    using iterator = std::move_iterator<cpp20_input_iterator<int*>>;

    LIBCPP_STATIC_ASSERT(!std::default_initializable<iterator>);
    static_assert(!std::copyable<iterator>);
    static_assert( std::input_iterator<iterator>);
    static_assert(!std::forward_iterator<iterator>);
    static_assert(!std::sentinel_for<iterator, iterator>); // not copyable
    static_assert(!std::sized_sentinel_for<iterator, iterator>);
    static_assert(!std::indirectly_movable<int*, iterator>);
    static_assert(!std::indirectly_movable_storable<int*, iterator>);
    static_assert(!std::indirectly_copyable<int*, iterator>);
    static_assert(!std::indirectly_copyable_storable<int*, iterator>);
    static_assert( std::indirectly_readable<iterator>);
    static_assert(!std::indirectly_writable<iterator, int>);
    static_assert( std::indirectly_swappable<iterator, iterator>);
  }
  {
    using iterator = std::move_iterator<forward_iterator<int*>>;

    static_assert( std::default_initializable<iterator>);
    static_assert( std::copyable<iterator>);
    static_assert( std::forward_iterator<iterator>);
    static_assert(!std::bidirectional_iterator<iterator>);
    static_assert( std::sentinel_for<iterator, iterator>);
    static_assert(!std::sized_sentinel_for<iterator, iterator>);
    static_assert(!std::indirectly_movable<int*, iterator>);
    static_assert(!std::indirectly_movable_storable<int*, iterator>);
    static_assert(!std::indirectly_copyable<int*, iterator>);
    static_assert(!std::indirectly_copyable_storable<int*, iterator>);
    static_assert( std::indirectly_readable<iterator>);
    static_assert(!std::indirectly_writable<iterator, int>);
    static_assert( std::indirectly_swappable<iterator, iterator>);
  }
  {
    using iterator = std::move_iterator<bidirectional_iterator<int*>>;

    static_assert( std::default_initializable<iterator>);
    static_assert( std::copyable<iterator>);
    static_assert( std::bidirectional_iterator<iterator>);
    static_assert(!std::random_access_iterator<iterator>);
    static_assert( std::sentinel_for<iterator, iterator>);
    static_assert(!std::sized_sentinel_for<iterator, iterator>);
    static_assert(!std::indirectly_movable<int*, iterator>);
    static_assert(!std::indirectly_movable_storable<int*, iterator>);
    static_assert(!std::indirectly_copyable<int*, iterator>);
    static_assert(!std::indirectly_copyable_storable<int*, iterator>);
    static_assert( std::indirectly_readable<iterator>);
    static_assert(!std::indirectly_writable<iterator, int>);
    static_assert( std::indirectly_swappable<iterator, iterator>);
  }
  {
    using iterator = std::move_iterator<random_access_iterator<int*>>;

    static_assert( std::default_initializable<iterator>);
    static_assert( std::copyable<iterator>);
    static_assert( std::random_access_iterator<iterator>);
    static_assert(!std::contiguous_iterator<iterator>);
    static_assert( std::sentinel_for<iterator, iterator>);
    static_assert( std::sized_sentinel_for<iterator, iterator>);
    static_assert(!std::indirectly_movable<int*, iterator>);
    static_assert(!std::indirectly_movable_storable<int*, iterator>);
    static_assert(!std::indirectly_copyable<int*, iterator>);
    static_assert(!std::indirectly_copyable_storable<int*, iterator>);
    static_assert( std::indirectly_readable<iterator>);
    static_assert(!std::indirectly_writable<iterator, int>);
    static_assert( std::indirectly_swappable<iterator, iterator>);
  }
  {
    using iterator = std::move_iterator<contiguous_iterator<int*>>;

    static_assert( std::default_initializable<iterator>);
    static_assert( std::copyable<iterator>);
    static_assert( std::random_access_iterator<iterator>);
    static_assert(!std::contiguous_iterator<iterator>);
    static_assert( std::sentinel_for<iterator, iterator>);
    static_assert( std::sized_sentinel_for<iterator, iterator>);
    static_assert(!std::indirectly_movable<int*, iterator>);
    static_assert(!std::indirectly_movable_storable<int*, iterator>);
    static_assert(!std::indirectly_copyable<int*, iterator>);
    static_assert(!std::indirectly_copyable_storable<int*, iterator>);
    static_assert( std::indirectly_readable<iterator>);
    static_assert(!std::indirectly_writable<iterator, int>);
    static_assert( std::indirectly_swappable<iterator, iterator>);
  }
  {
    using iterator = std::move_iterator<int*>;

    static_assert( std::default_initializable<iterator>);
    static_assert( std::copyable<iterator>);
    static_assert( std::random_access_iterator<iterator>);
    static_assert(!std::contiguous_iterator<iterator>);
    static_assert( std::sentinel_for<iterator, iterator>);
    static_assert( std::sized_sentinel_for<iterator, iterator>);
    static_assert(!std::indirectly_movable<int*, iterator>);
    static_assert(!std::indirectly_movable_storable<int*, iterator>);
    static_assert(!std::indirectly_copyable<int*, iterator>);
    static_assert(!std::indirectly_copyable_storable<int*, iterator>);
    static_assert( std::indirectly_readable<iterator>);
    static_assert(!std::indirectly_writable<iterator, int>);
    static_assert( std::indirectly_swappable<iterator, iterator>);
  }
}
