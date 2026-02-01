//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// std::equal(segmented-iterator, segmented-iterator, other-iterator);
// std::equal(other-iterator, other-iterator, segmented-iterator);
// std::equal(segmented-iterator, segmented-iterator, segmented-iterator);

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <algorithm>
#include <cassert>
#include <concepts>
#include <deque>
#include <iterator>
#include <vector>

template <class InContainer1, class InContainer2>
constexpr void test_containers() {
  using InIter1 = typename InContainer1::iterator;
  using InIter2 = typename InContainer2::iterator;

  { // Simple positive test
    InContainer1 lhs{1, 2, 3, 4};
    InContainer2 rhs{1, 2, 3, 4};

    std::same_as<bool> auto ret = std::equal(InIter1(lhs.begin()), InIter1(lhs.end()), InIter2(rhs.begin()));
    assert(ret);
  }
  { // Simple negative test
    InContainer1 lhs{1, 2, 3, 4};
    InContainer2 rhs{2, 2, 3, 4};

    std::same_as<bool> auto ret = std::equal(InIter1(lhs.begin()), InIter1(lhs.end()), InIter2(rhs.begin()));
    assert(!ret);
  }
  { // Multiple segmentes are iterated, equal
    InContainer1 lhs;
    std::generate_n(std::back_inserter(lhs), 4095, [i = 0]() mutable { return i++; });
    InContainer2 rhs(lhs.begin(), lhs.end());

    assert(std::equal(InIter1(lhs.begin()), InIter1(lhs.end()), InIter2(rhs.begin())));
  }
  { // Multiple segmentes are iterated, not equal
    InContainer1 lhs;
    std::generate_n(std::back_inserter(lhs), 4095, [i = 0]() mutable { return i++; });
    InContainer2 rhs(lhs.begin(), lhs.end());
    rhs.back() = -1;

    assert(!std::equal(InIter1(lhs.begin()), InIter1(lhs.end()), InIter2(rhs.begin())));
  }
  { // Multiple segmentes are iterated, first segment is partially iterated, equal
    InContainer1 lhs;
    std::generate_n(std::back_inserter(lhs), 4095, [i = 0]() mutable { return i++; });
    InContainer2 rhs;
    rhs.resize(10);
    rhs.insert(rhs.end(), lhs.begin(), lhs.end());
    rhs.erase(rhs.begin(), rhs.begin() + 10);

    assert(std::equal(InIter1(lhs.begin()), InIter1(lhs.end()), InIter2(rhs.begin())));
  }
  { // Multiple segmentes are iterated, first segment is partially iterated, not equal
    InContainer1 lhs;
    std::generate_n(std::back_inserter(lhs), 4095, [i = 0]() mutable { return i++; });
    InContainer2 rhs;
    rhs.resize(10);
    rhs.insert(rhs.end(), lhs.begin(), lhs.end());
    rhs.erase(rhs.begin(), rhs.begin() + 10);
    rhs.back() = -1;

    assert(!std::equal(InIter1(lhs.begin()), InIter1(lhs.end()), InIter2(rhs.begin())));
  }
}

int main(int, char**) {
  test_containers<std::vector<int>, std::deque<int>>();
  test_containers<std::deque<int>, std::vector<int>>();
  test_containers<std::deque<int>, std::deque<int>>();

  return 0;
}
