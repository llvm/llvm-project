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

// ranges::advance

#include <iterator>

#include <array>
#include <cassert>

#include "test_standard_function.h"
#include "test_iterators.h"

static_assert(is_function_like<decltype(std::ranges::advance)>());

using range_t = std::array<int, 10>;

[[nodiscard]] constexpr bool operator==(output_iterator<int*> const x, output_iterator<int*> const y) {
  return x.base() == y.base();
}

template <std::input_or_output_iterator I>
constexpr void check_round_trip(stride_counting_iterator<I> const& i, std::ptrdiff_t const n) {
  auto const distance = n < 0 ? -n : n;
  assert(i.stride_count() == distance);
  assert(i.stride_displacement() == n);
}

template <std::random_access_iterator I>
constexpr void check_round_trip(stride_counting_iterator<I> const& i, std::ptrdiff_t const n) {
  assert(i.stride_count() == 0 || i.stride_count() == 1);
  assert(i.stride_displacement() == n < 0 ? -1 : 1);
}

namespace iterator_count {
template <std::input_or_output_iterator I>
constexpr void check_move_forward(std::ptrdiff_t const n) {
  auto range = range_t{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  auto first = stride_counting_iterator(I(range.begin()));
  std::ranges::advance(first, n);
  assert(std::move(first).base().base() == range.begin() + n);
  check_round_trip(first, n);
}

template <std::bidirectional_iterator I>
constexpr void check_move_backward(std::ptrdiff_t const n) {
  auto range = range_t{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  auto first = stride_counting_iterator(I(range.begin() + n));
  std::ranges::advance(first, -n);
  assert(std::move(first).base().base() == range.begin());
  check_round_trip(first, -n);
}

[[nodiscard]] constexpr bool test() {
  check_move_forward<cpp17_input_iterator<range_t::const_iterator> >(1);
  check_move_forward<cpp20_input_iterator<range_t::const_iterator> >(2);
  check_move_forward<forward_iterator<range_t::const_iterator> >(3);
  check_move_forward<bidirectional_iterator<range_t::const_iterator> >(4);
  check_move_forward<random_access_iterator<range_t::const_iterator> >(5);
  check_move_forward<contiguous_iterator<range_t::const_iterator> >(6);
  check_move_forward<output_iterator<range_t::iterator> >(7);

  check_move_backward<bidirectional_iterator<range_t::const_iterator> >(4);
  check_move_backward<random_access_iterator<range_t::const_iterator> >(5);
  check_move_backward<contiguous_iterator<range_t::const_iterator> >(6);

  // Zero should be checked for each case and each overload
  check_move_forward<cpp17_input_iterator<range_t::const_iterator> >(0);
  check_move_forward<cpp20_input_iterator<range_t::const_iterator> >(0);
  check_move_forward<forward_iterator<range_t::const_iterator> >(0);
  check_move_forward<bidirectional_iterator<range_t::const_iterator> >(0);
  check_move_forward<random_access_iterator<range_t::const_iterator> >(0);
  check_move_forward<output_iterator<range_t::iterator> >(0);
  check_move_backward<bidirectional_iterator<range_t::const_iterator> >(0);
  check_move_backward<random_access_iterator<range_t::const_iterator> >(0);

  return true;
}
} // namespace iterator_count

class distance_apriori_sentinel {
public:
  distance_apriori_sentinel() = default;
  constexpr explicit distance_apriori_sentinel(std::ptrdiff_t const count) : count_(count) {}

  [[nodiscard]] constexpr bool operator==(std::input_or_output_iterator auto const&) const {
    assert(false && "difference op should take precedence");
    return false;
  }

  [[nodiscard]] constexpr friend std::ptrdiff_t operator-(std::input_or_output_iterator auto const&,
                                                          distance_apriori_sentinel const y) {
    return -y.count_;
  }

  [[nodiscard]] constexpr friend std::ptrdiff_t operator-(distance_apriori_sentinel const x,
                                                          std::input_or_output_iterator auto const&) {
    return x.count_;
  }

private:
  std::ptrdiff_t count_ = 0;
};

namespace iterator_sentinel {
template <std::input_or_output_iterator I, std::sentinel_for<I> S = I>
constexpr void check_assignable_case(std::ptrdiff_t const n) {
  auto range = range_t{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  auto first = stride_counting_iterator(I(range.begin()));
  std::ranges::advance(first, stride_counting_iterator(S(I(range.begin() + n))));
  assert(std::move(first).base().base() == range.begin() + n);
  assert(first.stride_count() == 0); // always zero, so don't use `check_round_trip`
}

template <std::input_or_output_iterator I>
constexpr void check_sized_sentinel_case(std::ptrdiff_t const n) {
  auto range = range_t{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  auto first = stride_counting_iterator(I(range.begin()));
  std::ranges::advance(first, distance_apriori_sentinel(n));
  assert(std::move(first).base().base() == range.begin() + n);
  check_round_trip(first, n);
}

template <std::input_or_output_iterator I>
constexpr void check_sentinel_case(std::ptrdiff_t const n) {
  auto range = range_t{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  auto first = stride_counting_iterator(I(range.begin()));
  auto const last = I(range.begin() + n);
  std::ranges::advance(first, sentinel_wrapper(last));
  assert(first.base() == last);
  assert(first.stride_count() == n); // always `n`, so don't use `check_round_trip`
}

[[nodiscard]] constexpr bool test() {
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
} // namespace iterator_sentinel

namespace iterator_count_sentinel {
struct expected_t {
  range_t::const_iterator coordinate;
  std::ptrdiff_t result;
};

template <std::input_or_output_iterator I>
constexpr void check_forward_sized_sentinel_case(std::ptrdiff_t const n, expected_t const expected, range_t& range) {
  auto current = stride_counting_iterator(I(range.begin()));
  auto const result = std::ranges::advance(current, n, distance_apriori_sentinel(range.size()));
  assert(current.base().base() == expected.coordinate);
  assert(result == expected.result);
  check_round_trip(current, n - expected.result);
}

template <std::random_access_iterator I>
constexpr void check_backward_sized_sentinel_case(std::ptrdiff_t const n, expected_t const expected, range_t& range) {
  auto current = stride_counting_iterator(I(range.end()));
  auto const result = std::ranges::advance(current, -n, stride_counting_iterator(I(range.begin())));
  assert(current.base().base() == expected.coordinate);
  assert(result == expected.result);
  check_round_trip(current, n - expected.result);
}

template <std::input_or_output_iterator I>
constexpr void check_forward_case(std::ptrdiff_t const n, expected_t const expected, range_t& range) {
  auto current = stride_counting_iterator(I(range.begin()));
  auto const result = std::ranges::advance(current, n, sentinel_wrapper(I(range.end())));
  assert(current.base().base() == expected.coordinate);
  assert(result == expected.result);
  assert(current.stride_count() == n - expected.result);
}

template <std::bidirectional_iterator I>
constexpr void check_backward_case(std::ptrdiff_t const n, expected_t const expected, range_t& range) {
  auto current = stride_counting_iterator(I(range.end()));
  auto const result = std::ranges::advance(current, -n, stride_counting_iterator(I(range.begin())));
  assert(current.base().base() == expected.coordinate);
  assert(result == expected.result);
  assert(current.stride_count() == n + expected.result);
  assert(current.stride_count() == -current.stride_displacement());
}

[[nodiscard]] constexpr bool test() {
  auto range = range_t{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  check_forward_sized_sentinel_case<cpp17_input_iterator<range_t::const_iterator> >(1, {range.begin() + 1, 0}, range);
  // cpp20_input_iterator not copyable, so is omitted
  check_forward_sized_sentinel_case<forward_iterator<range_t::const_iterator> >(3, {range.begin() + 3, 0}, range);
  check_forward_sized_sentinel_case<bidirectional_iterator<range_t::const_iterator> >(4, {range.begin() + 4, 0}, range);
  check_forward_sized_sentinel_case<random_access_iterator<range_t::const_iterator> >(5, {range.begin() + 5, 0}, range);
  check_forward_sized_sentinel_case<contiguous_iterator<range_t::const_iterator> >(6, {range.begin() + 6, 0}, range);
  check_forward_sized_sentinel_case<output_iterator<range_t::iterator> >(7, {range.begin() + 7, 0}, range);

  // bidirectional_iterator omitted because `n < 0` case requires `same_as<I, S>`
  check_backward_sized_sentinel_case<random_access_iterator<range_t::const_iterator> >(5, {range.begin() + 5, 0},
                                                                                       range);
  check_backward_sized_sentinel_case<contiguous_iterator<range_t::const_iterator> >(6, {range.begin() + 4, 0}, range);

  // disntance == range.size()
  check_forward_sized_sentinel_case<forward_iterator<range_t::const_iterator> >(10, {range.end(), 0}, range);
  check_forward_sized_sentinel_case<output_iterator<range_t::iterator> >(10, {range.end(), 0}, range);
  check_backward_sized_sentinel_case<random_access_iterator<range_t::const_iterator> >(10, {range.begin(), 0}, range);

  // distance > range.size()
  check_forward_sized_sentinel_case<forward_iterator<range_t::const_iterator> >(1000, {range.end(), 990}, range);
  check_forward_sized_sentinel_case<output_iterator<range_t::iterator> >(1000, {range.end(), 990}, range);
  check_backward_sized_sentinel_case<random_access_iterator<range_t::const_iterator> >(1000, {range.begin(), -990},
                                                                                       range);

  check_forward_case<cpp17_input_iterator<range_t::const_iterator> >(1, {range.begin() + 1, 0}, range);
  check_forward_case<forward_iterator<range_t::const_iterator> >(3, {range.begin() + 3, 0}, range);
  check_forward_case<bidirectional_iterator<range_t::const_iterator> >(4, {range.begin() + 4, 0}, range);
  check_forward_case<random_access_iterator<range_t::const_iterator> >(5, {range.begin() + 5, 0}, range);
  check_forward_case<contiguous_iterator<range_t::const_iterator> >(6, {range.begin() + 6, 0}, range);
  check_forward_case<output_iterator<range_t::iterator> >(7, {range.begin() + 7, 0}, range);
  check_backward_case<bidirectional_iterator<range_t::const_iterator> >(8, {range.begin() + 2, 0}, range);

  // disntance == range.size()
  check_forward_case<forward_iterator<range_t::const_iterator> >(10, {range.end(), 0}, range);
  check_forward_case<output_iterator<range_t::iterator> >(10, {range.end(), 0}, range);
  check_backward_case<bidirectional_iterator<range_t::const_iterator> >(10, {range.begin(), 0}, range);

  // distance > range.size()
  check_forward_case<forward_iterator<range_t::const_iterator> >(1000, {range.end(), 990}, range);
  check_forward_case<output_iterator<range_t::iterator> >(1000, {range.end(), 990}, range);
  check_backward_case<bidirectional_iterator<range_t::const_iterator> >(1000, {range.begin(), -990}, range);

  return true;
}
} // namespace iterator_count_sentinel

int main(int, char**) {
  static_assert(iterator_count::test());
  assert(iterator_count::test());

  static_assert(iterator_sentinel::test());
  assert(iterator_sentinel::test());

  static_assert(iterator_count_sentinel::test());
  assert(iterator_count_sentinel::test());

  return 0;
}
