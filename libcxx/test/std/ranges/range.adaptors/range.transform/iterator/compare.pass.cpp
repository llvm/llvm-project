//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// transform_view::<iterator>::operator{<,>,<=,>=,==,!=,<=>}

#include <ranges>
#include <compare>

#include "test_macros.h"
#include "test_iterators.h"
#include "../types.h"

constexpr bool test() {
  {
    // Test a new-school iterator with operator<=>; the transform iterator should also have operator<=>.
    using It = three_way_contiguous_iterator<int*>;
    static_assert(std::three_way_comparable<It>);
    using R = std::ranges::transform_view<std::ranges::subrange<It>, PlusOne>;
    static_assert(std::three_way_comparable<std::ranges::iterator_t<R>>);

    int a[] = {1,2,3};
    std::same_as<R> auto r = std::ranges::subrange<It>(It(a), It(a+3)) | std::views::transform(PlusOne());
    auto iter1 = r.begin();
    auto iter2 = iter1 + 1;

    assert(!(iter1 < iter1));  assert(iter1 < iter2);     assert(!(iter2 < iter1));
    assert(iter1 <= iter1);    assert(iter1 <= iter2);    assert(!(iter2 <= iter1));
    assert(!(iter1 > iter1));  assert(!(iter1 > iter2));  assert(iter2 > iter1);
    assert(iter1 >= iter1);    assert(!(iter1 >= iter2)); assert(iter2 >= iter1);
    assert(iter1 == iter1);    assert(!(iter1 == iter2)); assert(iter2 == iter2);
    assert(!(iter1 != iter1)); assert(iter1 != iter2);    assert(!(iter2 != iter2));

    assert((iter1 <=> iter2) == std::strong_ordering::less);
    assert((iter1 <=> iter1) == std::strong_ordering::equal);
    assert((iter2 <=> iter1) == std::strong_ordering::greater);
  }

  {
    // Test an old-school iterator with no operator<=>; the transform iterator shouldn't have operator<=> either.
    using It = random_access_iterator<int*>;
    static_assert(!std::three_way_comparable<It>);
    using R = std::ranges::transform_view<std::ranges::subrange<It>, PlusOne>;
    static_assert(!std::three_way_comparable<std::ranges::iterator_t<R>>);

    int a[] = {1,2,3};
    std::same_as<R> auto r = std::ranges::subrange<It>(It(a), It(a+3)) | std::views::transform(PlusOne());
    auto iter1 = r.begin();
    auto iter2 = iter1 + 1;

    assert(!(iter1 < iter1));  assert(iter1 < iter2);     assert(!(iter2 < iter1));
    assert(iter1 <= iter1);    assert(iter1 <= iter2);    assert(!(iter2 <= iter1));
    assert(!(iter1 > iter1));  assert(!(iter1 > iter2));  assert(iter2 > iter1);
    assert(iter1 >= iter1);    assert(!(iter1 >= iter2)); assert(iter2 >= iter1);
    assert(iter1 == iter1);    assert(!(iter1 == iter2)); assert(iter2 == iter2);
    assert(!(iter1 != iter1)); assert(iter1 != iter2);    assert(!(iter2 != iter2));
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
