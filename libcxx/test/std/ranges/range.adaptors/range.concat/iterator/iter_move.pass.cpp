//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// friend constexpr range_rvalue_reference_t<V> iter_move(iterator const& i)
//  noexcept(noexcept(ranges::iter_move(i.current_)));

#include <ranges>

#include <array>
#include <cassert>
#include <utility>
#include "test_iterators.h"
#include "test_macros.h"
#include "../types.h"

struct Range : std::ranges::view_base {
  using Iterator = forward_iterator<int*>;
  using Sentinel = sentinel_wrapper<Iterator>;
  constexpr explicit Range(int* b, int* e) : begin_(b), end_(e) { }
  constexpr Iterator begin() const { return Iterator(begin_); }
  constexpr Sentinel end() const { return Sentinel(Iterator(end_)); }

private:
  int* begin_;
  int* end_;
};

template <class Iterator, bool HasNoexceptIterMove>
constexpr bool test() {
  using Sentinel = sentinel_wrapper<Iterator>;
  using View = minimal_view<Iterator, Sentinel>;
  using ConcatView = std::ranges::concat_view<View>;
  using ConcatIterator = std::ranges::iterator_t<ConcatView>;

  auto make_concat_view = [](auto begin, auto end) {
    View view{Iterator(begin), Sentinel(Iterator(end))};
    return ConcatView(std::move(view));
  };

  // noexcept in case of a forward iterator
  {
    int buff[] = {0, 1, 2, 3};
    ConcatView view = make_concat_view(buff, buff + 4);
    ConcatIterator it = view.begin();
    int&& result = iter_move(it);
    static_assert(noexcept(iter_move(it)) == HasNoexceptIterMove);
    assert(&result == buff);
  }

  return true;
}

constexpr bool tests() {
   test<cpp17_input_iterator<int*>,           /* noexcept */ false>();
   test<forward_iterator<int*>,               /* noexcept */ false>();
   test<bidirectional_iterator<int*>,         /* noexcept */ false>();
   test<random_access_iterator<int*>,         /* noexcept */ false>();
   test<contiguous_iterator<int*>,            /* noexcept */ false>();
   test<int*,                                 /* noexcept */ true>();
  return true;
}

int main(int, char**) {
  tests();
  static_assert(tests());
  return 0;
}
