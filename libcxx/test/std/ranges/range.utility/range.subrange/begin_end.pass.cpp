//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// class std::ranges::subrange;

#include <ranges>
#include <cassert>
#include <iterator>

#include "test_iterators.h"
#include "type_algorithms.h"

template <class Iterator, class Sentinel>
constexpr bool test() {
  using Subrange = std::ranges::subrange<Iterator, Sentinel>;

  // Empty subrange
  {
    int array[10] = {};
    Subrange rng(Iterator(std::begin(array)), Sentinel(Iterator(std::begin(array))));
    std::same_as<Iterator> decltype(auto) beg = rng.begin();
    std::same_as<Sentinel> decltype(auto) end = rng.end();
    assert(beg == Iterator(std::begin(array)));
    assert(end == Iterator(std::begin(array)));
  }

  // Non-empty subrange
  {
    int array[10] = {};
    Subrange rng(Iterator(std::begin(array)), Sentinel(Iterator(std::end(array) - 3)));
    std::same_as<Iterator> decltype(auto) beg = rng.begin();
    std::same_as<Sentinel> decltype(auto) end = rng.end();
    assert(beg == Iterator(std::begin(array)));
    assert(end == Iterator(std::end(array) - 3));
  }

  return true;
}

int main(int, char**) {
  types::for_each(types::forward_iterator_list<int*>{}, []<class Iterator> {
    test<Iterator, sentinel_wrapper<Iterator>>();
    static_assert(test<Iterator, sentinel_wrapper<Iterator>>());
  });

  return 0;
}
