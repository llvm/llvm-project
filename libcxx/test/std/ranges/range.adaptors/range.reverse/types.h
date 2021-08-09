//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_REVERSE_TYPES_H
#define TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_REVERSE_TYPES_H

#include "test_macros.h"
#include "test_iterators.h"

struct BidirRange : std::ranges::view_base {
  int *ptr_;

  constexpr BidirRange(int *ptr) : ptr_(ptr) {}

  constexpr bidirectional_iterator<int*> begin() { return bidirectional_iterator<int*>{ptr_}; }
  constexpr bidirectional_iterator<const int*> begin() const { return bidirectional_iterator<const int*>{ptr_}; }
  constexpr bidirectional_iterator<int*> end() { return bidirectional_iterator<int*>{ptr_ + 8}; }
  constexpr bidirectional_iterator<const int*> end() const { return bidirectional_iterator<const int*>{ptr_ + 8}; }
};

enum CopyCategory { MoveOnly, Copyable };
template<CopyCategory CC>
struct BidirSentRange : std::ranges::view_base {
  using sent_t = sentinel_wrapper<bidirectional_iterator<int*>>;
  using sent_const_t = sentinel_wrapper<bidirectional_iterator<const int*>>;

  int *ptr_;

  constexpr BidirSentRange(int *ptr) : ptr_(ptr) {}
  constexpr BidirSentRange(const BidirSentRange &) requires (CC == Copyable) = default;
  constexpr BidirSentRange(BidirSentRange &&) requires (CC == MoveOnly) = default;
  constexpr BidirSentRange& operator=(const BidirSentRange &) requires (CC == Copyable) = default;
  constexpr BidirSentRange& operator=(BidirSentRange &&) requires (CC == MoveOnly) = default;

  constexpr bidirectional_iterator<int*> begin() { return bidirectional_iterator<int*>{ptr_}; }
  constexpr bidirectional_iterator<const int*> begin() const { return bidirectional_iterator<const int*>{ptr_}; }
  constexpr sent_t end() { return sent_t{bidirectional_iterator<int*>{ptr_ + 8}}; }
  constexpr sent_const_t end() const { return sent_const_t{bidirectional_iterator<const int*>{ptr_ + 8}}; }
};

#endif // TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_REVERSE_TYPES_H
