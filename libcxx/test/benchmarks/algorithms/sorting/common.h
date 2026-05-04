//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LIBCXX_TEST_BENCHMARKS_ALGORITHMS_SORTING_COMMON_H
#define LIBCXX_TEST_BENCHMARKS_ALGORITHMS_SORTING_COMMON_H

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <random>
#include <type_traits>
#include <vector>

namespace support {

// This function creates a vector with N int-like values.
//
// These values are arranged in such a way that they would invoke O(N^2)
// behavior on any quick sort implementation that satisifies certain conditions.
// Details are available in the following paper:
//
//  "A Killer Adversary for Quicksort", M. D. McIlroy, Software-Practice &
//  Experience Volume 29 Issue 4 April 10, 1999 pp 341-344.
//  https://dl.acm.org/doi/10.5555/311868.311871.
template <class T>
std::vector<T> quicksort_adversarial_data(std::size_t n) {
  static_assert(std::is_integral_v<T>);
  assert(n > 0);

  // If an element is equal to gas, it indicates that the value of the element
  // is still to be decided and may change over the course of time.
  T gas = n - 1;

  std::vector<T> v;
  v.resize(n);
  for (unsigned int i = 0; i < n; ++i) {
    v[i] = gas;
  }
  // Candidate for the pivot position.
  int candidate = 0;
  int nsolid    = 0;
  // Populate all positions in the generated input to gas.
  std::vector<int> ascending_values(v.size());

  // Fill up with ascending values from 0 to v.size()-1.  These will act as
  // indices into v.
  std::iota(ascending_values.begin(), ascending_values.end(), 0);
  std::sort(ascending_values.begin(), ascending_values.end(), [&](int x, int y) {
    if (v[x] == gas && v[y] == gas) {
      // We are comparing two inputs whose value is still to be decided.
      if (x == candidate) {
        v[x] = nsolid++;
      } else {
        v[y] = nsolid++;
      }
    }
    if (v[x] == gas) {
      candidate = x;
    } else if (v[y] == gas) {
      candidate = y;
    }
    return v[x] < v[y];
  });
  return v;
}

// ascending sorted values
template <class T>
std::vector<T> ascending_sorted_data(std::size_t n) {
  std::vector<T> v(n);
  std::iota(v.begin(), v.end(), 0);
  return v;
}

// descending sorted values
template <class T>
std::vector<T> descending_sorted_data(std::size_t n) {
  std::vector<T> v(n);
  std::iota(v.begin(), v.end(), 0);
  std::reverse(v.begin(), v.end());
  return v;
}

// pipe-organ pattern
template <class T>
std::vector<T> pipe_organ_data(std::size_t n) {
  std::vector<T> v(n);
  std::iota(v.begin(), v.end(), 0);
  auto half = v.begin() + v.size() / 2;
  std::reverse(half, v.end());
  return v;
}

// heap pattern
template <class T>
std::vector<T> heap_data(std::size_t n) {
  std::vector<T> v(n);
  std::iota(v.begin(), v.end(), 0);
  std::make_heap(v.begin(), v.end());
  return v;
}

// shuffled randomly
template <class T>
std::vector<T> shuffled_data(std::size_t n) {
  std::vector<T> v(n);
  std::iota(v.begin(), v.end(), 0);
  std::mt19937 rng;
  std::shuffle(v.begin(), v.end(), rng);
  return v;
}

// single element in the whole sequence
template <class T>
std::vector<T> single_element_data(std::size_t n) {
  std::vector<T> v(n);
  return v;
}

struct NonIntegral {
  NonIntegral() : value_(0) {}
  NonIntegral(int i) : value_(i) {}
  friend auto operator<(NonIntegral const& a, NonIntegral const& b) { return a.value_ < b.value_; }
  friend auto operator>(NonIntegral const& a, NonIntegral const& b) { return a.value_ > b.value_; }
  friend auto operator<=(NonIntegral const& a, NonIntegral const& b) { return a.value_ <= b.value_; }
  friend auto operator>=(NonIntegral const& a, NonIntegral const& b) { return a.value_ >= b.value_; }
  friend auto operator==(NonIntegral const& a, NonIntegral const& b) { return a.value_ == b.value_; }
  friend auto operator!=(NonIntegral const& a, NonIntegral const& b) { return a.value_ != b.value_; }

private:
  int value_;
};

} // namespace support

#endif // LIBCXX_TEST_BENCHMARKS_ALGORITHMS_SORTING_COMMON_H
