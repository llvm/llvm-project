//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts

// ranges::advance(it, sent)

#include <iterator>

#include <array>
#include <cassert>
#include <cstddef>

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

template <class It, class Sent = It>
constexpr void check_assignable_case() {
  auto range = range_t{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  for (std::ptrdiff_t n = 0; n != 9; ++n) {
    {
      It first(range.begin());
      Sent last(It(range.begin() + n));
      std::ranges::advance(first, last);
      assert(base(first) == range.begin() + n);
    }

    // Count operations
    if constexpr (std::is_same_v<It, Sent>) {
      stride_counting_iterator<It> first(It(range.begin()));
      stride_counting_iterator<It> last(It(range.begin() + n));
      std::ranges::advance(first, last);
      assert(first.base().base() == range.begin() + n);
      assert(first.stride_count() == 0); // because we got here by assigning from last, not by incrementing
    }
  }
}

template <class It>
constexpr void check_sized_sentinel_case() {
  auto range = range_t{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  for (std::ptrdiff_t n = 0; n != 9; ++n) {
    {
      It first(range.begin());
      distance_apriori_sentinel last(n);
      std::ranges::advance(first, last);
      assert(base(first) == range.begin() + n);
    }

    // Count operations
    {
      stride_counting_iterator<It> first(It(range.begin()));
      distance_apriori_sentinel last(n);
      std::ranges::advance(first, last);

      assert(first.base().base() == range.begin() + n);
      if constexpr (std::random_access_iterator<It>) {
        assert(first.stride_count() == 1);
        assert(first.stride_displacement() == 1);
      } else {
        assert(first.stride_count() == n);
        assert(first.stride_displacement() == n);
      }
    }
  }
}

template <class It>
constexpr void check_sentinel_case() {
  auto range = range_t{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  for (std::ptrdiff_t n = 0; n != 9; ++n) {
    {
      It first(range.begin());
      sentinel_wrapper<It> last(It(range.begin() + n));
      std::ranges::advance(first, last);
      assert(base(first) == range.begin() + n);
    }

    // Count operations
    {
      stride_counting_iterator<It> first(It(range.begin()));
      sentinel_wrapper<It> last(It(range.begin() + n));
      std::ranges::advance(first, last);
      assert(first.base() == last);
      assert(first.stride_count() == n);
    }
  }
}

constexpr bool test() {
  using It = range_t::const_iterator;
  check_assignable_case<cpp17_input_iterator<It>, sentinel_wrapper<cpp17_input_iterator<It>>>();
  check_assignable_case<forward_iterator<It>>();
  check_assignable_case<bidirectional_iterator<It>>();
  check_assignable_case<random_access_iterator<It>>();
  check_assignable_case<contiguous_iterator<It>>();

  check_sized_sentinel_case<cpp17_input_iterator<It>>();
  check_sized_sentinel_case<cpp20_input_iterator<It>>();
  check_sized_sentinel_case<forward_iterator<It>>();
  check_sized_sentinel_case<bidirectional_iterator<It>>();
  check_sized_sentinel_case<random_access_iterator<It>>();
  check_sized_sentinel_case<contiguous_iterator<It>>();

  check_sentinel_case<cpp17_input_iterator<It>>();
  // cpp20_input_iterator not copyable, so is omitted
  check_sentinel_case<forward_iterator<It>>();
  check_sentinel_case<bidirectional_iterator<It>>();
  check_sentinel_case<random_access_iterator<It>>();
  check_sentinel_case<contiguous_iterator<It>>();
  return true;
}

int main(int, char**) {
  assert(test());
  static_assert(test());
  return 0;
}
