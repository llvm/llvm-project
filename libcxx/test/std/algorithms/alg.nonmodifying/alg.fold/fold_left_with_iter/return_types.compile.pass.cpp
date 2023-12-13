//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// Checks that `std::ranges::fold_left_with_iter`'s return type is correct.

#include <algorithm>
#include <concepts>
#include <functional>
#include <ranges>

#include "test_iterators.h"
#include "test_range.h"

template <class Result, class Range, class T>
concept is_in_value_result = std::same_as<Result, std::ranges::in_value_result<std::ranges::iterator_t<Range>, T>>;

template <class Result, class T>
concept is_dangling_with = std::same_as<Result, std::ranges::in_value_result<std::ranges::dangling, T>>;

using std::ranges::fold_left_with_iter;
[[maybe_unused]] auto f = [](int x, double y) { return x * y; };

struct Int {
  int value;
};

struct Long {
  int value;

  Long plus(Int) const;
};

namespace sentinel_based_ranges {
template <class T>
using cpp17_input_range = test_range<cpp17_input_iterator, T>;

static_assert(requires(cpp17_input_range<int> r) {
  { fold_left_with_iter(r.begin(), r.end(), 0, std::plus()) } -> is_in_value_result<cpp17_input_range<int>, int>;
});
static_assert(requires(cpp17_input_range<int> r) {
  { fold_left_with_iter(r, 0, std::plus()) } -> is_in_value_result<cpp17_input_range<int>, int>;
});
static_assert(requires(cpp17_input_range<int> r) {
  { fold_left_with_iter(std::move(r), 0, std::plus()) } -> is_dangling_with<int>;
});

template <class T>
using cpp20_input_range = test_range<cpp20_input_iterator, T>;

static_assert(requires(cpp20_input_range<int> r) {
  { fold_left_with_iter(r.begin(), r.end(), 0, std::plus()) } -> is_in_value_result<cpp20_input_range<int>, int>;
});
static_assert(requires(cpp20_input_range<int> r) {
  { fold_left_with_iter(r, 0, std::plus()) } -> is_in_value_result<cpp20_input_range<int>, int>;
});
static_assert(requires(cpp20_input_range<int> r) {
  { fold_left_with_iter(std::move(r), 0, std::plus()) } -> is_dangling_with<int>;
});

template <class T>
using forward_range = test_range<forward_iterator, T>;

static_assert(requires(forward_range<int> r) {
  { fold_left_with_iter(r.begin(), r.end(), 0, std::plus()) } -> is_in_value_result<forward_range<int>, int>;
});
static_assert(requires(forward_range<int> r) {
  { fold_left_with_iter(r, 0, std::plus()) } -> is_in_value_result<forward_range<int>, int>;
});
static_assert(requires(forward_range<int> r) {
  { fold_left_with_iter(std::move(r), 0, std::plus()) } -> is_dangling_with<int>;
});

template <class T>
using bidirectional_range = test_range<bidirectional_iterator, T>;

static_assert(requires(bidirectional_range<short> r) {
  { fold_left_with_iter(r.begin(), r.end(), 0, f) } -> is_in_value_result<bidirectional_range<short>, double>;
});
static_assert(requires(bidirectional_range<short> r) {
  { fold_left_with_iter(r, 0, f) } -> is_in_value_result<bidirectional_range<short>, double>;
});
static_assert(requires(bidirectional_range<short> r) {
  { fold_left_with_iter(std::move(r), 0, f) } -> is_dangling_with<double>;
});

template <class T>
using random_access_range = test_range<random_access_iterator, T>;

static_assert(requires(random_access_range<int> r) {
  { fold_left_with_iter(r.begin(), r.end(), 0, f) } -> is_in_value_result<random_access_range<int>, double>;
});
static_assert(requires(random_access_range<int> r) {
  { fold_left_with_iter(r, 0, f) } -> is_in_value_result<random_access_range<int>, double>;
});
static_assert(requires(random_access_range<int> r) {
  { fold_left_with_iter(std::move(r), 0, f) } -> is_dangling_with<double>;
});

template <class T>
using contiguous_range = test_range<contiguous_iterator, T>;

static_assert(requires(contiguous_range<int> r) {
  { fold_left_with_iter(r.begin(), r.end(), 0.0f, std::plus()) } -> is_in_value_result<contiguous_range<int>, float>;
});
static_assert(requires(contiguous_range<int> r) {
  { fold_left_with_iter(r, 0.0f, std::plus()) } -> is_in_value_result<contiguous_range<int>, float>;
});
static_assert(requires(contiguous_range<int> r) {
  { fold_left_with_iter(std::move(r), 0.0f, std::plus()) } -> is_dangling_with<float>;
});
} // namespace sentinel_based_ranges

namespace common_ranges {
template <class T>
using forward_range = test_common_range<forward_iterator, T>;

static_assert(requires(forward_range<Int> r) {
  { fold_left_with_iter(r.begin(), r.end(), Long(0), &Long::plus) } -> is_in_value_result<forward_range<Int>, Long>;
});
static_assert(requires(forward_range<Int> r) {
  { fold_left_with_iter(r, Long(0), &Long::plus) } -> is_in_value_result<forward_range<Int>, Long>;
});
static_assert(requires(forward_range<Int> r) {
  { fold_left_with_iter(std::move(r), Long(0), &Long::plus) } -> is_dangling_with<Long>;
});

template <class T>
using bidirectional_range = test_common_range<bidirectional_iterator, T>;

static_assert(requires(bidirectional_range<int> r) {
  { fold_left_with_iter(r.begin(), r.end(), 0, std::plus()) } -> is_in_value_result<bidirectional_range<int>, int>;
});
static_assert(requires(bidirectional_range<int> r) {
  { fold_left_with_iter(r, 0, std::plus()) } -> is_in_value_result<bidirectional_range<int>, int>;
});
static_assert(requires(bidirectional_range<int> r) {
  { fold_left_with_iter(std::move(r), 0, std::plus()) } -> is_dangling_with<int>;
});

template <class T>
using random_access_range = test_common_range<random_access_iterator, T>;

static_assert(requires(random_access_range<int> r) {
  { fold_left_with_iter(r.begin(), r.end(), 0, f) } -> is_in_value_result<random_access_range<int>, double>;
});
static_assert(requires(random_access_range<int> r) {
  { fold_left_with_iter(r, 0, f) } -> is_in_value_result<random_access_range<int>, double>;
});
static_assert(requires(random_access_range<int> r) {
  { fold_left_with_iter(std::move(r), 0, f) } -> is_dangling_with<double>;
});

template <class T>
using contiguous_range = test_common_range<contiguous_iterator, T>;

static_assert(requires(contiguous_range<int> r) {
  { fold_left_with_iter(r.begin(), r.end(), 0.0f, std::plus()) } -> is_in_value_result<contiguous_range<int>, float>;
});
static_assert(requires(contiguous_range<int> r) {
  { fold_left_with_iter(r, 0.0f, std::plus()) } -> is_in_value_result<contiguous_range<int>, float>;
});
static_assert(requires(contiguous_range<int> r) {
  { fold_left_with_iter(std::move(r), 0.0f, std::plus()) } -> is_dangling_with<float>;
});
} // namespace common_ranges
