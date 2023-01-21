//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <algorithm>
#include <array>
#include <cassert>
#include <concepts>
#include <deque>
#include <ranges>
#include <vector>

#include "test_iterators.h"
#include "type_algorithms.h"

template <class InContainer, class OutContainer>
constexpr void test_containers() {
  using InIter  = typename InContainer::iterator;
  using OutIter = typename OutContainer::iterator;

  {
    InContainer in{1, 2, 3, 4};
    OutContainer out(4);

    std::same_as<std::ranges::in_out_result<InIter, OutIter>> auto ret =
        std::ranges::copy(in.begin(), in.end(), out.begin());
    assert(std::ranges::equal(in, out));
    assert(ret.in == in.end());
    assert(ret.out == out.end());
  }
  {
    InContainer in{1, 2, 3, 4};
    OutContainer out(4);
    std::same_as<std::ranges::in_out_result<InIter, OutIter>> auto ret = std::ranges::copy(in, out.begin());
    assert(std::ranges::equal(in, out));
    assert(ret.in == in.end());
    assert(ret.out == out.end());
  }
}

template <class Iter, class Sent>
constexpr void test_join_view() {
  auto to_subranges = std::views::transform([](auto& vec) {
    return std::ranges::subrange(Iter(vec.data()), Sent(Iter(vec.data() + vec.size())));
  });

  { // segmented -> contiguous
    std::vector<std::vector<int>> vectors = {};
    auto range                            = vectors | to_subranges;
    std::vector<std::ranges::subrange<Iter, Sent>> subrange_vector(range.begin(), range.end());
    std::array<int, 0> arr;

    std::ranges::copy(subrange_vector | std::views::join, arr.begin());
    assert(std::ranges::equal(arr, std::array<int, 0>{}));
  }
  { // segmented -> contiguous
    std::vector<std::vector<int>> vectors = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10}, {}};
    auto range                            = vectors | to_subranges;
    std::vector<std::ranges::subrange<Iter, Sent>> subrange_vector(range.begin(), range.end());
    std::array<int, 10> arr;

    std::ranges::copy(subrange_vector | std::views::join, arr.begin());
    assert(std::ranges::equal(arr, std::array{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}));
  }
  { // contiguous -> segmented
    std::vector<std::vector<int>> vectors = {{0, 0, 0, 0}, {0, 0}, {0, 0, 0, 0}, {}};
    auto range                            = vectors | to_subranges;
    std::vector<std::ranges::subrange<Iter, Sent>> subrange_vector(range.begin(), range.end());
    std::array arr = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    std::ranges::copy(arr, (subrange_vector | std::views::join).begin());
    assert(std::ranges::equal(subrange_vector | std::views::join, std::array{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}));
  }
  { // segmented -> segmented
    std::vector<std::vector<int>> vectors = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10}, {}};
    auto range1                           = vectors | to_subranges;
    std::vector<std::ranges::subrange<Iter, Sent>> subrange_vector(range1.begin(), range1.end());
    std::vector<std::vector<int>> to_vectors = {{0, 0, 0, 0}, {0, 0, 0, 0}, {}, {0, 0}};
    auto range2                              = to_vectors | to_subranges;
    std::vector<std::ranges::subrange<Iter, Sent>> to_subrange_vector(range2.begin(), range2.end());

    std::ranges::copy(subrange_vector | std::views::join, (to_subrange_vector | std::views::join).begin());
    assert(std::ranges::equal(to_subrange_vector | std::views::join, std::array{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}));
  }
}

int main(int, char**) {
  if (!std::is_constant_evaluated()) {
    test_containers<std::deque<int>, std::deque<int>>();
    test_containers<std::deque<int>, std::vector<int>>();
    test_containers<std::vector<int>, std::deque<int>>();
    test_containers<std::vector<int>, std::vector<int>>();
  }

  meta::for_each(meta::forward_iterator_list<int*>{}, []<class Iter> {
    test_join_view<Iter, Iter>();
    test_join_view<Iter, sentinel_wrapper<Iter>>();
    test_join_view<Iter, sized_sentinel<Iter>>();
  });

  return 0;
}
