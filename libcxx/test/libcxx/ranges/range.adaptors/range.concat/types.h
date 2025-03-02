//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_LIBCXX_RANGES_RANGE_ADAPTORS_RANGE_CONCAT_TYPES_H
#define TEST_LIBCXX_RANGES_RANGE_ADAPTORS_RANGE_CONCAT_TYPES_H

#include <ranges>
#include <utility>

int globalArray[8] = {0, 1, 2, 3, 4, 5, 6, 7};

struct ThrowOnCopyView : std::ranges::view_base {
  int start_;
  int* ptr_;
  constexpr explicit ThrowOnCopyView(int* ptr = globalArray, int start = 0) : start_(start), ptr_(ptr) {}
  constexpr ThrowOnCopyView(ThrowOnCopyView&&) = default;
  constexpr ThrowOnCopyView(const ThrowOnCopyView&) { throw 42; };
  constexpr ThrowOnCopyView& operator=(ThrowOnCopyView&&) = default;
  constexpr ThrowOnCopyView& operator=(const ThrowOnCopyView&) {
    throw 42;
    return *this;
  };
  constexpr int* begin() const { return ptr_ + start_; }
  constexpr int* end() const { return ptr_ + 8; }
};

#endif // TEST_LIBCXX_RANGES_RANGE_ADAPTORS_CONCAT_FILTER_TYPES_H