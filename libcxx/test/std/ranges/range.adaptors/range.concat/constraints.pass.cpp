//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

#include <cassert>
#include <list>
#include <ranges>
#include <vector>
#include "test_iterators.h"
#include "test_macros.h"

// This tests https://cplusplus.github.io/LWG/issue4082
// views::concat(r) is well-formed when r is an output_range

template<typename T>
concept WellFormedView = requires(T& a) {
  std::views::concat(a);
};

struct InputRange {
  using Iterator = cpp17_input_iterator<int*>;
  using Sentinel = sentinel_wrapper<Iterator>;
  constexpr InputRange(int* b, int *e): begin_(b), end_(e) {}
  constexpr Iterator begin() { return Iterator(begin_); }
  constexpr Sentinel end() { return Sentinel(Iterator(end_)); }

private:
  int* begin_;
  int* end_;
};


int main(int, char**) {

  // rejects when it is an output range
  {
    std::vector<int> v{1,2,3};
    static_assert(!WellFormedView<decltype(std::views::counted(std::back_inserter(v), 3))>);
  }

  // input range
  {
    static_assert(WellFormedView<InputRange>);
  }

  // bidirectional range
  {
    static_assert(WellFormedView<std::list<int>>);
  }

  // random access range
  {
    static_assert(WellFormedView<std::vector<int>>);
  }

  return 0;
}
