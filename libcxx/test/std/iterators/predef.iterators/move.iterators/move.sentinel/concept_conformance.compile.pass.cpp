//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <iterator>

// template<semiregular S>
//   class move_sentinel;

#include <iterator>

#include "test_iterators.h"

void test()
{
  // Pointer.
  {
    using It = int*;
    static_assert( std::sentinel_for<std::move_sentinel<It>, std::move_iterator<It>>);
    static_assert( std::sized_sentinel_for<std::move_sentinel<It>, std::move_iterator<It>>);
    static_assert( std::sentinel_for<std::move_sentinel<sentinel_wrapper<It>>, std::move_iterator<It>>);
    static_assert(!std::sized_sentinel_for<std::move_sentinel<sentinel_wrapper<It>>, std::move_iterator<It>>);
    static_assert( std::sentinel_for<std::move_sentinel<sized_sentinel<It>>, std::move_iterator<It>>);
    static_assert( std::sized_sentinel_for<std::move_sentinel<sized_sentinel<It>>, std::move_iterator<It>>);
  }

  // `Cpp17InputIterator`.
  {
    using It = cpp17_input_iterator<int*>;
    static_assert( std::sentinel_for<std::move_sentinel<sentinel_wrapper<It>>, std::move_iterator<It>>);
    static_assert(!std::sized_sentinel_for<std::move_sentinel<sentinel_wrapper<It>>, std::move_iterator<It>>);
    static_assert( std::sentinel_for<std::move_sentinel<sized_sentinel<It>>, std::move_iterator<It>>);
    static_assert( std::sized_sentinel_for<std::move_sentinel<sized_sentinel<It>>, std::move_iterator<It>>);
  }

  // `std::input_iterator`.
  {
    using It = cpp20_input_iterator<int*>;
    static_assert( std::sentinel_for<std::move_sentinel<sentinel_wrapper<It>>, std::move_iterator<It>>);
    static_assert(!std::sized_sentinel_for<std::move_sentinel<sentinel_wrapper<It>>, std::move_iterator<It>>);
    static_assert( std::sentinel_for<std::move_sentinel<sized_sentinel<It>>, std::move_iterator<It>>);
    static_assert( std::sized_sentinel_for<std::move_sentinel<sized_sentinel<It>>, std::move_iterator<It>>);
  }

  // `std::forward_iterator`.
  {
    using It = forward_iterator<int*>;
    static_assert( std::sentinel_for<std::move_sentinel<It>, std::move_iterator<It>>);
    static_assert(!std::sized_sentinel_for<std::move_sentinel<It>, std::move_iterator<It>>);
    static_assert( std::sentinel_for<std::move_sentinel<sentinel_wrapper<It>>, std::move_iterator<It>>);
    static_assert(!std::sized_sentinel_for<std::move_sentinel<sentinel_wrapper<It>>, std::move_iterator<It>>);
    static_assert( std::sentinel_for<std::move_sentinel<sized_sentinel<It>>, std::move_iterator<It>>);
    static_assert( std::sized_sentinel_for<std::move_sentinel<sized_sentinel<It>>, std::move_iterator<It>>);
  }

  // `std::bidirectional_iterator`.
  {
    using It = bidirectional_iterator<int*>;
    static_assert( std::sentinel_for<std::move_sentinel<It>, std::move_iterator<It>>);
    static_assert(!std::sized_sentinel_for<std::move_sentinel<It>, std::move_iterator<It>>);
    static_assert( std::sentinel_for<std::move_sentinel<sentinel_wrapper<It>>, std::move_iterator<It>>);
    static_assert(!std::sized_sentinel_for<std::move_sentinel<sentinel_wrapper<It>>, std::move_iterator<It>>);
    static_assert( std::sentinel_for<std::move_sentinel<sized_sentinel<It>>, std::move_iterator<It>>);
    static_assert( std::sized_sentinel_for<std::move_sentinel<sized_sentinel<It>>, std::move_iterator<It>>);
  }

  // `std::random_access_iterator`.
  {
    using It = random_access_iterator<int*>;
    static_assert( std::sentinel_for<std::move_sentinel<It>, std::move_iterator<It>>);
    static_assert( std::sized_sentinel_for<std::move_sentinel<It>, std::move_iterator<It>>);
    static_assert( std::sentinel_for<std::move_sentinel<sentinel_wrapper<It>>, std::move_iterator<It>>);
    static_assert(!std::sized_sentinel_for<std::move_sentinel<sentinel_wrapper<It>>, std::move_iterator<It>>);
    static_assert( std::sentinel_for<std::move_sentinel<sized_sentinel<It>>, std::move_iterator<It>>);
    static_assert( std::sized_sentinel_for<std::move_sentinel<sized_sentinel<It>>, std::move_iterator<It>>);
  }

  // `std::contiguous_iterator`.
  {
    using It = contiguous_iterator<int*>;
    static_assert( std::sentinel_for<std::move_sentinel<It>, std::move_iterator<It>>);
    static_assert( std::sized_sentinel_for<std::move_sentinel<It>, std::move_iterator<It>>);
    static_assert( std::sentinel_for<std::move_sentinel<sentinel_wrapper<It>>, std::move_iterator<It>>);
    static_assert(!std::sized_sentinel_for<std::move_sentinel<sentinel_wrapper<It>>, std::move_iterator<It>>);
    static_assert( std::sentinel_for<std::move_sentinel<sized_sentinel<It>>, std::move_iterator<It>>);
    static_assert( std::sized_sentinel_for<std::move_sentinel<sized_sentinel<It>>, std::move_iterator<It>>);
  }

  // `std::contiguous_iterator` with the spaceship operator.
  {
    using It = three_way_contiguous_iterator<int*>;
    static_assert( std::sentinel_for<std::move_sentinel<It>, std::move_iterator<It>>);
    static_assert( std::sized_sentinel_for<std::move_sentinel<It>, std::move_iterator<It>>);
    static_assert( std::sentinel_for<std::move_sentinel<sentinel_wrapper<It>>, std::move_iterator<It>>);
    static_assert(!std::sized_sentinel_for<std::move_sentinel<sentinel_wrapper<It>>, std::move_iterator<It>>);
    static_assert( std::sentinel_for<std::move_sentinel<sized_sentinel<It>>, std::move_iterator<It>>);
    static_assert( std::sized_sentinel_for<std::move_sentinel<sized_sentinel<It>>, std::move_iterator<It>>);
  }
}
