//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_ADAPTOR_TYPES_H
#define TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_ADAPTOR_TYPES_H

#include <cstdint>
#include <ranges>

#include "test_macros.h"
#include "test_iterators.h"

#if TEST_STD_VER <= 20
#  error "range.adaptor/types.h" can only be included in builds supporting C++20
#endif // TEST_STD_VER <= 20

struct ForwardSizedNonCommon {
  int* buffer_      = nullptr;
  std::size_t size_ = 0;

  template <std::size_t N>
  constexpr ForwardSizedNonCommon(int (&b)[N]) : buffer_(b), size_(N) {}

  constexpr ForwardSizedNonCommon(int* b, std::size_t s) : buffer_(b), size_(s) {}

  using iterator = forward_sized_iterator<int*>;
  using sentinel = sized_sentinel<iterator>;

  constexpr iterator begin() const { return iterator(buffer_); }
  constexpr sentinel end() const { return sentinel(iterator(buffer_ + size_)); }
};
static_assert(std::ranges::forward_range<ForwardSizedNonCommon>);
static_assert(std::ranges::sized_range<ForwardSizedNonCommon>);
static_assert(!std::ranges::common_range<ForwardSizedNonCommon>);
static_assert(!std::ranges::random_access_range<ForwardSizedNonCommon>);

#endif //  TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_ADAPTOR_TYPES_H
