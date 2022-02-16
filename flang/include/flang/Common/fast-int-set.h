//===-- include/flang/Common/fast-int-set.h --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Implements a Briggs-Torczon fast set of integers in a fixed small range
// [0..(n-1)] This is a data structure with no dynamic memory allocation and all
// O(1) elemental operations.  It does not need to initialize its internal state
// arrays, but you can call its InitializeState() member function to avoid
// complaints from valgrind.

// The set is implemented with two arrays and an element count.
// 1) The distinct values in the set occupy the leading elements of
//    value_[0 .. size_-1] in arbitrary order.  Their positions may change
//    when other values are removed from the set with Remove().
// 2) For 0 <= j < size_, index_[value_[j]] == j.
// 3) If only Add() and PopValue() are used, the popped values will be the
//    most recently Add()ed distinct unpopped values; i.e., the value_ array
//    will function as a stack whose top is at (size_-1).

#ifndef FORTRAN_COMMON_FAST_INT_SET_H_
#define FORTRAN_COMMON_FAST_INT_SET_H_

#include <optional>

namespace Fortran::common {

template <int N> class FastIntSet {
public:
  static_assert(N > 0);
  static constexpr int maxValue{N - 1};

  int size() const { return size_; }
  const int *value() const { return &value_[0]; }

  bool IsValidValue(int n) const { return n >= 0 && n <= maxValue; }

  void Clear() { size_ = 0; }

  bool IsEmpty() const { return size_ == 0; }

  void InitializeState() {
    if (!isFullyInitialized_) {
      for (int j{size_}; j < N; ++j) {
        value_[j] = index_[j] = 0;
      }
      isFullyInitialized_ = true;
    }
  }

  bool Contains(int n) const {
    if (IsValidValue(n)) {
      int j{index_[n]};
      return j >= 0 && j < size_ && value_[j] == n;
    } else {
      return false;
    }
  }

  bool Add(int n) {
    if (IsValidValue(n)) {
      if (!UncheckedContains(n)) {
        value_[index_[n] = size_++] = n;
      }
      return true;
    } else {
      return false;
    }
  }

  bool Remove(int n) {
    if (IsValidValue(n)) {
      if (UncheckedContains(n)) {
        int last{value_[--size_]};
        value_[index_[last] = index_[n]] = last;
      }
      return true;
    } else {
      return false;
    }
  }

  std::optional<int> PopValue() {
    if (IsEmpty()) {
      return std::nullopt;
    } else {
      return value_[--size_];
    }
  }

private:
  bool UncheckedContains(int n) const {
    int j{index_[n]};
    return j >= 0 && j < size_ && value_[j] == n;
  }

  int value_[N];
  int index_[N];
  int size_{0};
  bool isFullyInitialized_{false}; // memory was cleared
};
} // namespace Fortran::common
#endif // FORTRAN_COMMON_FAST_INT_SET_H_
