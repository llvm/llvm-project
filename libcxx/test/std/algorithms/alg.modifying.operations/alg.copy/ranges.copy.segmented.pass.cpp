//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <algorithm>
#include <cassert>
#include <concepts>
#include <deque>
#include <vector>

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

int main(int, char**) {
  if (!std::is_constant_evaluated()) {
    test_containers<std::deque<int>, std::deque<int>>();
    test_containers<std::deque<int>, std::vector<int>>();
    test_containers<std::vector<int>, std::deque<int>>();
    test_containers<std::vector<int>, std::vector<int>>();
  }

  return 0;
}
