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

// ranges::advance(it, n, sent)

#include <iterator>

#include <array>
#include <cassert>

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

struct expected_t {
  range_t::const_iterator coordinate;
  std::ptrdiff_t result;
};

template <std::input_or_output_iterator It>
constexpr void check_forward_sized_sentinel(std::ptrdiff_t n, expected_t expected, range_t& range) {
  using Difference = std::iter_difference_t<It>;
  auto current = stride_counting_iterator(It(range.begin()));
  Difference const result = std::ranges::advance(current, n, distance_apriori_sentinel(range.size()));
  assert(current.base().base() == expected.coordinate);
  assert(result == expected.result);

  if constexpr (std::random_access_iterator<It>) {
    assert(current.stride_count() == 0 || current.stride_count() == 1);
    assert(current.stride_displacement() == current.stride_count());
  } else {
    assert(current.stride_count() == (n - result));
    assert(current.stride_displacement() == (n - result));
  }
}

template <std::random_access_iterator It>
constexpr void check_backward_sized_sentinel(std::ptrdiff_t n, expected_t expected, range_t& range) {
  using Difference = std::iter_difference_t<It>;
  auto current = stride_counting_iterator(It(range.end()));
  Difference const result = std::ranges::advance(current, -n, stride_counting_iterator(It(range.begin())));
  assert(current.base().base() == expected.coordinate);
  assert(result == expected.result);

  assert(current.stride_count() == 0 || current.stride_count() == 1);
  assert(current.stride_displacement() == current.stride_count());
}

template <std::input_or_output_iterator It>
constexpr void check_forward(std::ptrdiff_t n, expected_t expected, range_t& range) {
  using Difference = std::iter_difference_t<It>;
  auto current = stride_counting_iterator(It(range.begin()));
  Difference const result = std::ranges::advance(current, n, sentinel_wrapper(It(range.end())));
  assert(current.base().base() == expected.coordinate);
  assert(result == expected.result);
  assert(current.stride_count() == n - result);
}

template <std::bidirectional_iterator It>
constexpr void check_backward(std::ptrdiff_t n, expected_t expected, range_t& range) {
  using Difference = std::iter_difference_t<It>;
  auto current = stride_counting_iterator(It(range.end()));
  Difference const result = std::ranges::advance(current, -n, stride_counting_iterator(It(range.begin())));
  assert(current.base().base() == expected.coordinate);
  assert(result == expected.result);
  assert(current.stride_count() == n + result);
  assert(current.stride_count() == -current.stride_displacement());
}

constexpr bool test() {
  auto range = range_t{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  check_forward_sized_sentinel<cpp17_input_iterator<range_t::const_iterator> >(1, {range.begin() + 1, 0}, range);
  // cpp20_input_iterator not copyable, so is omitted
  check_forward_sized_sentinel<forward_iterator<range_t::const_iterator> >(3, {range.begin() + 3, 0}, range);
  check_forward_sized_sentinel<bidirectional_iterator<range_t::const_iterator> >(4, {range.begin() + 4, 0}, range);
  check_forward_sized_sentinel<random_access_iterator<range_t::const_iterator> >(5, {range.begin() + 5, 0}, range);
  check_forward_sized_sentinel<contiguous_iterator<range_t::const_iterator> >(6, {range.begin() + 6, 0}, range);

  // bidirectional_iterator omitted because the `n < 0` case requires `same_as<I, S>`
  check_backward_sized_sentinel<random_access_iterator<range_t::const_iterator> >(5, {range.begin() + 5, 0}, range);
  check_backward_sized_sentinel<contiguous_iterator<range_t::const_iterator> >(6, {range.begin() + 4, 0}, range);

  // distance == range.size()
  check_forward_sized_sentinel<forward_iterator<range_t::const_iterator> >(10, {range.end(), 0}, range);
  check_backward_sized_sentinel<random_access_iterator<range_t::const_iterator> >(10, {range.begin(), 0}, range);

  // distance > range.size()
  check_forward_sized_sentinel<forward_iterator<range_t::const_iterator> >(1000, {range.end(), 990}, range);
  check_backward_sized_sentinel<random_access_iterator<range_t::const_iterator> >(1000, {range.begin(), -990}, range);

  check_forward<cpp17_input_iterator<range_t::const_iterator> >(1, {range.begin() + 1, 0}, range);
  check_forward<forward_iterator<range_t::const_iterator> >(3, {range.begin() + 3, 0}, range);
  check_forward<bidirectional_iterator<range_t::const_iterator> >(4, {range.begin() + 4, 0}, range);
  check_forward<random_access_iterator<range_t::const_iterator> >(5, {range.begin() + 5, 0}, range);
  check_forward<contiguous_iterator<range_t::const_iterator> >(6, {range.begin() + 6, 0}, range);
  check_backward<bidirectional_iterator<range_t::const_iterator> >(8, {range.begin() + 2, 0}, range);

  // distance == range.size()
  check_forward<forward_iterator<range_t::const_iterator> >(10, {range.end(), 0}, range);
  check_backward<bidirectional_iterator<range_t::const_iterator> >(10, {range.begin(), 0}, range);

  // distance > range.size()
  check_forward<forward_iterator<range_t::const_iterator> >(1000, {range.end(), 990}, range);
  check_backward<bidirectional_iterator<range_t::const_iterator> >(1000, {range.begin(), -990}, range);

  return true;
}

int main(int, char**) {
  assert(test());
  static_assert(test());
  return 0;
}
