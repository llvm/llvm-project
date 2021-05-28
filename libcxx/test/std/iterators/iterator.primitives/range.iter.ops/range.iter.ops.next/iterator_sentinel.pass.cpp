//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: gcc-10

// ranges::next

#include <iterator>

#include <array>
#include <cassert>

#include "check_round_trip.h"
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

  constexpr friend std::ptrdiff_t operator-(std::input_or_output_iterator auto const&,
                                            distance_apriori_sentinel const y) {
    return -y.count_;
  }

  constexpr friend std::ptrdiff_t operator-(distance_apriori_sentinel const x,
                                            std::input_or_output_iterator auto const&) {
    return x.count_;
  }

private:
  std::ptrdiff_t count_ = 0;
};

template <std::input_or_output_iterator I>
constexpr void check_assignable_case(std::ptrdiff_t const n) {
  auto range = range_t{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  auto result =
      std::ranges::next(stride_counting_iterator(I(range.begin())), stride_counting_iterator(I(range.begin() + n)));
  assert(result.base().base() == range.begin() + n);
  assert(result.stride_count() == 0); // always zero, so don't use `check_round_trip`
}

template <std::input_or_output_iterator I>
constexpr void check_sized_sentinel_case(std::ptrdiff_t const n) {
  auto range = range_t{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  auto result = std::ranges::next(stride_counting_iterator(I(range.begin())), distance_apriori_sentinel(n));
  assert(std::move(result).base().base() == range.begin() + n);
  check_round_trip(result, n);
}

template <std::input_or_output_iterator I>
constexpr void check_sentinel_case(std::ptrdiff_t const n) {
  auto range = range_t{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  auto const last = I(range.begin() + n);
  auto result = std::ranges::next(stride_counting_iterator(I(range.begin())), sentinel_wrapper(last));
  assert(std::move(result).base() == last);
  assert(result.stride_count() == n); // always `n`, so don't use `check_round_trip`
}

constexpr bool check_iterator_sentinel() {
  check_assignable_case<cpp17_input_iterator<range_t::const_iterator> >(1);
  check_assignable_case<forward_iterator<range_t::const_iterator> >(3);
  check_assignable_case<bidirectional_iterator<range_t::const_iterator> >(4);
  check_assignable_case<random_access_iterator<range_t::const_iterator> >(5);
  check_assignable_case<contiguous_iterator<range_t::const_iterator> >(6);
  check_assignable_case<output_iterator<range_t::iterator> >(7);

  check_sized_sentinel_case<cpp17_input_iterator<range_t::const_iterator> >(7);
  check_sized_sentinel_case<cpp20_input_iterator<range_t::const_iterator> >(6);
  check_sized_sentinel_case<forward_iterator<range_t::const_iterator> >(5);
  check_sized_sentinel_case<bidirectional_iterator<range_t::const_iterator> >(4);
  check_sized_sentinel_case<random_access_iterator<range_t::const_iterator> >(3);
  check_sized_sentinel_case<contiguous_iterator<range_t::const_iterator> >(2);
  check_sized_sentinel_case<output_iterator<range_t::iterator> >(1);

  check_sentinel_case<cpp17_input_iterator<range_t::const_iterator> >(1);
  // cpp20_input_iterator not copyable, so is omitted
  check_sentinel_case<forward_iterator<range_t::const_iterator> >(3);
  check_sentinel_case<bidirectional_iterator<range_t::const_iterator> >(4);
  check_sentinel_case<random_access_iterator<range_t::const_iterator> >(5);
  check_sentinel_case<contiguous_iterator<range_t::const_iterator> >(6);
  check_sentinel_case<output_iterator<range_t::iterator> >(7);
  return true;
}

int main(int, char**) {
  static_assert(check_iterator_sentinel());
  check_iterator_sentinel();
  return 0;
}
