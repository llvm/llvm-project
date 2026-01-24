//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++17

// UNSUPPORTED: libcpp-has-no-incomplete-pstl

// Checks the logic of the index wrapper iterator.

#include <algorithm>
#include <execution>
#include <iterator>
#include <type_traits>
#include <cassert>
#include <cstddef>

int main(int, char**) {
  using Index = std::ptrdiff_t;
  using Iter  = std::__pstl::__index_iterator<Index>;
  static_assert(std::is_nothrow_default_constructible_v<Iter>);
  static_assert(std::is_nothrow_constructible_v<Iter, Index>);
  static_assert(std::is_trivially_copy_constructible_v<Iter>);
  static_assert(std::is_trivially_move_constructible_v<Iter>);
  static_assert(std::is_trivially_assignable_v<Iter, Iter>);
  static_assert(std::is_trivially_copy_assignable_v<Iter>);
  static_assert(std::is_trivially_move_assignable_v<Iter>);
  static_assert(std::is_trivially_destructible_v<Iter>);

  {
    assert((*Iter{}) == Index{});
  }
  {
    assert((*Iter{7}) == 7);
  }
  {
    Iter it{5};
    assert(&(++it) == &it);
    assert((*it) == 6);
  }
  {
    Iter it{5};
    assert(*(it++) == 5);
    assert((*it) == 6);
  }
  {
    Iter it{5};
    assert(&(--it) == &it);
    assert((*it) == 4);
  }
  {
    Iter it{5};
    assert(*(it--) == 5);
    assert((*it) == 4);
  }
  {
    assert(*(Iter{4} + 3) == 7);
  }
  {
    assert(*(4 + Iter{3}) == 7);
  }
  {
    Iter it{5};
    assert(&(it += 3) == &it);
    assert((*it) == 8);
  }
  {
    assert((Iter{4} - 3) == Iter{1});
  }
  {
    assert((Iter{4} - Iter{3}) == 1);
  }
  {
    Iter it{5};
    assert(&(it -= 3) == &it);
    assert((*it) == 2);
  }
  {
    assert(Iter{5}[3] == 8);
  }
  {
    assert(Iter{5} == Iter{5});
    assert(!(Iter{5} == Iter{6}));
  }
  {
    assert(Iter{5} != Iter{6});
    assert(!(Iter{5} != Iter{5}));
  }

  return 0;
}
