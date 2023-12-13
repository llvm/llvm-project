//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// Checks that `std::ranges::fold_left`'s return type is correct.

#include <algorithm>
#include <concepts>
#include <functional>
#include <ranges>

#include "test_iterators.h"
#include "test_range.h"

template <class Result, class T>
concept is_T = std::same_as<Result, T>;

using std::ranges::fold_left;
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
  { fold_left(r.begin(), r.end(), 0, std::plus()) } -> std::same_as<int>;
});
static_assert(requires(cpp17_input_range<int> r) {
  { fold_left(r, 0, std::plus()) } -> std::same_as<int>;
});
static_assert(requires(cpp17_input_range<int> r) {
  { fold_left(std::move(r), 0, std::plus()) } -> std::same_as<int>;
});

template <class T>
using cpp20_input_range = test_range<cpp20_input_iterator, T>;

static_assert(requires(cpp20_input_range<int> r) {
  { fold_left(r.begin(), r.end(), 0, std::plus()) } -> std::same_as<int>;
});
static_assert(requires(cpp20_input_range<int> r) {
  { fold_left(r, 0, std::plus()) } -> std::same_as<int>;
});
static_assert(requires(cpp20_input_range<int> r) {
  { fold_left(std::move(r), 0, std::plus()) } -> std::same_as<int>;
});

template <class T>
using forward_range = test_range<forward_iterator, T>;

static_assert(requires(forward_range<int> r) {
  { fold_left(r.begin(), r.end(), 0, std::plus()) } -> std::same_as<int>;
});
static_assert(requires(forward_range<int> r) {
  { fold_left(r, 0, std::plus()) } -> std::same_as<int>;
});
static_assert(requires(forward_range<int> r) {
  { fold_left(std::move(r), 0, std::plus()) } -> std::same_as<int>;
});

template <class T>
using bidirectional_range = test_range<bidirectional_iterator, T>;

static_assert(requires(bidirectional_range<short> r) {
  { fold_left(r.begin(), r.end(), 0, f) } -> std::same_as<double>;
});
static_assert(requires(bidirectional_range<short> r) {
  { fold_left(r, 0, f) } -> std::same_as<double>;
});
static_assert(requires(bidirectional_range<short> r) {
  { fold_left(std::move(r), 0, f) } -> std::same_as<double>;
});

template <class T>
using random_access_range = test_range<random_access_iterator, T>;

static_assert(requires(random_access_range<int> r) {
  { fold_left(r.begin(), r.end(), 0, f) } -> std::same_as<double>;
});
static_assert(requires(random_access_range<int> r) {
  { fold_left(r, 0, f) } -> std::same_as<double>;
});
static_assert(requires(random_access_range<int> r) {
  { fold_left(std::move(r), 0, f) } -> std::same_as<double>;
});

template <class T>
using contiguous_range = test_range<contiguous_iterator, T>;

static_assert(requires(contiguous_range<int> r) {
  { fold_left(r.begin(), r.end(), 0.0f, std::plus()) } -> std::same_as<float>;
});
static_assert(requires(contiguous_range<int> r) {
  { fold_left(r, 0.0f, std::plus()) } -> std::same_as<float>;
});
static_assert(requires(contiguous_range<int> r) {
  { fold_left(std::move(r), 0.0f, std::plus()) } -> std::same_as<float>;
});
} // namespace sentinel_based_ranges

namespace common_ranges {
template <class T>
using forward_range = test_common_range<forward_iterator, T>;

static_assert(requires(forward_range<Int> r) {
  { fold_left(r.begin(), r.end(), Long(0), &Long::plus) } -> std::same_as<Long>;
});
static_assert(requires(forward_range<Int> r) {
  { fold_left(r, Long(0), &Long::plus) } -> std::same_as<Long>;
});
static_assert(requires(forward_range<Int> r) {
  { fold_left(std::move(r), Long(0), &Long::plus) } -> std::same_as<Long>;
});

template <class T>
using bidirectional_range = test_common_range<bidirectional_iterator, T>;

static_assert(requires(bidirectional_range<int> r) {
  { fold_left(r.begin(), r.end(), 0, std::plus()) } -> std::same_as<int>;
});
static_assert(requires(bidirectional_range<int> r) {
  { fold_left(r, 0, std::plus()) } -> std::same_as<int>;
});
static_assert(requires(bidirectional_range<int> r) {
  { fold_left(std::move(r), 0, std::plus()) } -> std::same_as<int>;
});

template <class T>
using random_access_range = test_common_range<random_access_iterator, T>;

static_assert(requires(random_access_range<int> r) {
  { fold_left(r.begin(), r.end(), 0, f) } -> std::same_as<double>;
});
static_assert(requires(random_access_range<int> r) {
  { fold_left(r, 0, f) } -> std::same_as<double>;
});
static_assert(requires(random_access_range<int> r) {
  { fold_left(std::move(r), 0, f) } -> std::same_as<double>;
});

template <class T>
using contiguous_range = test_common_range<contiguous_iterator, T>;

static_assert(requires(contiguous_range<int> r) {
  { fold_left(r.begin(), r.end(), 0.0f, std::plus()) } -> std::same_as<float>;
});
static_assert(requires(contiguous_range<int> r) {
  { fold_left(r, 0.0f, std::plus()) } -> std::same_as<float>;
});
static_assert(requires(contiguous_range<int> r) {
  { fold_left(std::move(r), 0.0f, std::plus()) } -> std::same_as<float>;
});
} // namespace common_ranges
