//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// constexpr iterator& operator++();
// constexpr iterator operator++(int);

#include <cassert>
#include <ranges>

#include "test_iterators.h"

template <class Iter>
constexpr void testOne() {
  constexpr auto make_subrange = []<std::size_t N>(int(&buffer)[N]) {
    return std::ranges::subrange<Iter>{Iter{buffer}, Iter{buffer + N}};
  };

  // next subrange does not reach the end
  {
    int buffer[] = {0, 1, 2, -1, 4, 5, -1, 7};
    auto input   = make_subrange(buffer);
    std::ranges::split_view sv(input, -1);
    using SplitIter = std::ranges::iterator_t<decltype(sv)>;

    // ++it
    {
      auto it = sv.begin();

      decltype(auto) it1 = ++it;
      static_assert(std::is_same_v<decltype(it1), SplitIter&>);
      assert(&it1 == &it);

      assert(base((*it).begin()) == buffer + 4);
      assert(base((*it).end()) == buffer + 6);

      ++it;
      assert(base((*it).begin()) == buffer + 7);
      assert(base((*it).end()) == buffer + 8);
    }

    // it++
    {
      auto it       = sv.begin();
      auto original = it;

      decltype(auto) it1 = it++;
      static_assert(std::is_same_v<decltype(it1), SplitIter>);
      assert(it1 == original);

      assert(base((*it).begin()) == buffer + 4);
      assert(base((*it).end()) == buffer + 6);

      it++;
      assert(base((*it).begin()) == buffer + 7);
      assert(base((*it).end()) == buffer + 8);
    }
  }

  // next's begin is the end
  {
    int buffer[] = {0, 1, 2};
    auto input   = make_subrange(buffer);
    std::ranges::split_view sv(input, -1);
    using SplitIter = std::ranges::iterator_t<decltype(sv)>;

    // ++it
    {
      auto it = sv.begin();

      decltype(auto) it1 = ++it; // trailing_empty is false
      static_assert(std::is_same_v<decltype(it1), SplitIter&>);
      assert(&it1 == &it);

      assert(it == sv.end());
    }

    // it++
    {
      auto it       = sv.begin();
      auto original = it;

      decltype(auto) it1 = it++; // trailing_empty is false
      static_assert(std::is_same_v<decltype(it1), SplitIter>);
      assert(it1 == original);

      assert(it == sv.end());
    }
  }

  // next's end is the end
  {
    int buffer[] = {0, 1, 2, -1};
    auto input   = make_subrange(buffer);
    std::ranges::split_view sv(input, -1);
    using SplitIter = std::ranges::iterator_t<decltype(sv)>;

    // ++it
    {
      auto it = sv.begin();

      decltype(auto) it1 = ++it; // trailing_empty is true
      static_assert(std::is_same_v<decltype(it1), SplitIter&>);
      assert(&it1 == &it);

      assert(it != sv.end());
      assert(base((*it).begin()) == buffer + 4);
      assert(base((*it).begin()) == buffer + 4);

      ++it;
      assert(it == sv.end());
    }

    // it++
    {
      auto it       = sv.begin();
      auto original = it;

      decltype(auto) it1 = it++; // trailing_empty is true
      static_assert(std::is_same_v<decltype(it1), SplitIter>);
      assert(it1 == original);

      assert(it != sv.end());

      assert(base((*it).begin()) == buffer + 4);
      assert(base((*it).begin()) == buffer + 4);

      it++;
      assert(it == sv.end());
    }
  }
}

constexpr bool test() {
  testOne<forward_iterator<int*>>();
  testOne<bidirectional_iterator<int*>>();
  testOne<random_access_iterator<int*>>();
  testOne<contiguous_iterator<int*>>();
  testOne<int*>();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
