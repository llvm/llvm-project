//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts

// ranges::next(it, bound)

#include <iterator>

#include <array>
#include <cassert>
#include <utility>

#include "test_iterators.h"

using range_t = std::array<int, 10>;

class distance_apriori_sentinel {
public:
  distance_apriori_sentinel() = default;
  constexpr explicit distance_apriori_sentinel(std::ptrdiff_t const count) : count_(count) {}

  constexpr bool operator==(std::input_or_output_iterator auto const&) const {
    assert(false && "difference op should take precedence");
    return false;
  }

  friend constexpr std::ptrdiff_t operator-(std::input_or_output_iterator auto const&,
                                            distance_apriori_sentinel const y) {
    return -y.count_;
  }

  friend constexpr std::ptrdiff_t operator-(distance_apriori_sentinel const x,
                                            std::input_or_output_iterator auto const&) {
    return x.count_;
  }

private:
  std::ptrdiff_t count_ = 0;
};

template <std::input_or_output_iterator It>
constexpr void check_assignable(It it, It last, int const* expected) {
  {
    It result = std::ranges::next(std::move(it), std::move(last));
    assert(&*result == expected);
  }

  // Count operations
  {
    auto strided_it = stride_counting_iterator(std::move(it));
    auto strided_last = stride_counting_iterator(std::move(last));
    auto result = std::ranges::next(std::move(strided_it), std::move(strided_last));
    assert(&*result == expected);
    assert(result.stride_count() == 0); // because we got here by assigning from last, not by incrementing
  }
}

template <std::input_or_output_iterator It>
constexpr void check_sized_sentinel(It it, It last, int const* expected) {
  auto n = (last.base() - it.base());

  {
    auto sent = distance_apriori_sentinel(n);
    auto result = std::ranges::next(std::move(it), sent);
    assert(&*result == expected);
  }

  // Count operations
  {
    auto strided_it = stride_counting_iterator(std::move(it));
    auto sent = distance_apriori_sentinel(n);
    auto result = std::ranges::next(std::move(strided_it), sent);
    assert(&*result == expected);

    if constexpr (std::random_access_iterator<It>) {
      assert(result.stride_count() == 1); // should have used exactly one +=
      assert(result.stride_displacement() == 1);
    } else {
      assert(result.stride_count() == n);
      assert(result.stride_displacement() == n);
    }
  }
}

template <std::input_or_output_iterator It>
constexpr void check_sentinel(It it, It last, int const* expected) {
  auto n = (last.base() - it.base());

  {
    auto sent = sentinel_wrapper(last);
    It result = std::ranges::next(std::move(it), sent);
    assert(&*result == expected);
  }

  // Count operations
  {
    auto strided_it = stride_counting_iterator(it);
    auto sent = sentinel_wrapper(stride_counting_iterator(last));
    stride_counting_iterator result = std::ranges::next(std::move(strided_it), sent);
    assert(&*result == expected);
    assert(result.stride_count() == n); // must have used ++ until it hit the sentinel
  }
}

constexpr bool test() {
  int range[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  check_assignable(cpp17_input_iterator(&range[0]), cpp17_input_iterator(&range[2]), &range[2]);
  check_assignable(forward_iterator(&range[0]), forward_iterator(&range[3]), &range[3]);
  check_assignable(bidirectional_iterator(&range[0]), bidirectional_iterator(&range[4]), &range[4]);
  check_assignable(random_access_iterator(&range[0]), random_access_iterator(&range[5]), &range[5]);
  check_assignable(contiguous_iterator(&range[0]), contiguous_iterator(&range[6]), &range[6]);

  check_sized_sentinel(cpp17_input_iterator(&range[0]), cpp17_input_iterator(&range[7]), &range[7]);
  check_sized_sentinel(cpp20_input_iterator(&range[0]), cpp20_input_iterator(&range[6]), &range[6]);
  check_sized_sentinel(forward_iterator(&range[0]), forward_iterator(&range[5]), &range[5]);
  check_sized_sentinel(bidirectional_iterator(&range[0]), bidirectional_iterator(&range[4]), &range[4]);
  check_sized_sentinel(random_access_iterator(&range[0]), random_access_iterator(&range[3]), &range[3]);
  check_sized_sentinel(contiguous_iterator(&range[0]), contiguous_iterator(&range[2]), &range[2]);

  check_sentinel(cpp17_input_iterator(&range[0]), cpp17_input_iterator(&range[1]), &range[1]);
  // cpp20_input_iterator not copyable, so is omitted
  check_sentinel(forward_iterator(&range[0]), forward_iterator(&range[3]), &range[3]);
  check_sentinel(bidirectional_iterator(&range[0]), bidirectional_iterator(&range[4]), &range[4]);
  check_sentinel(random_access_iterator(&range[0]), random_access_iterator(&range[5]), &range[5]);
  check_sentinel(contiguous_iterator(&range[0]), contiguous_iterator(&range[6]), &range[6]);
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
