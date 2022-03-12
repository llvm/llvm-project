//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template<class S, class I>
// concept sentinel_for;

#include <iterator>

static_assert(std::sentinel_for<int*, int*>);
static_assert(!std::sentinel_for<int*, long*>);
struct nth_element_sentinel {
  bool operator==(int*) const;
};
static_assert(std::sentinel_for<nth_element_sentinel, int*>);
static_assert(std::sentinel_for<nth_element_sentinel, int*>);

struct not_semiregular {
  not_semiregular() = delete;
  bool operator==(int*) const;
};
static_assert(!std::sentinel_for<not_semiregular, int*>);

struct weakly_equality_comparable_with_int {
  bool operator==(int) const;
};
static_assert(!std::sentinel_for<weakly_equality_comparable_with_int, int>);

struct move_only_iterator {
  using value_type = int;
  using difference_type = std::ptrdiff_t;

  move_only_iterator() = default;

  move_only_iterator(move_only_iterator&&) = default;
  move_only_iterator& operator=(move_only_iterator&&) = default;

  move_only_iterator(move_only_iterator const&) = delete;
  move_only_iterator& operator=(move_only_iterator const&) = delete;

  value_type operator*() const;
  move_only_iterator& operator++();
  move_only_iterator operator++(int);

  bool operator==(move_only_iterator const&) const = default;
};
static_assert(std::movable<move_only_iterator> && !std::copyable<move_only_iterator> &&
              std::input_or_output_iterator<move_only_iterator> &&
              !std::sentinel_for<move_only_iterator, move_only_iterator>);
