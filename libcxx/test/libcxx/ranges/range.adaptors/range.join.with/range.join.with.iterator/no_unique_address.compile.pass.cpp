//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <ranges>

// This test ensures that we use `[[no_unique_address]]` in `join_with_view::iterator`.

#include <cstddef>
#include <ranges>
#include <variant>

struct IntRange : std::ranges::view_base {
  int* begin();
  int* end();
};

class Iter {
public:
  using value_type      = IntRange;
  using difference_type = ptrdiff_t;

  Iter& operator++();
  void operator++(int);
  value_type& operator*() const;
  bool operator==(std::default_sentinel_t) const;

private:
  int* ptr_;
};

static_assert(std::input_iterator<Iter>);
static_assert(!std::forward_iterator<Iter>);

struct View : std::ranges::view_base {
  Iter begin();
  std::default_sentinel_t end();
};

static_assert(std::ranges::input_range<View>);
static_assert(!std::ranges::forward_range<View>);

using JWV = std::ranges::join_with_view<View, IntRange>;

// Expected JWV::iterator layout:
// _Parent* __parent_;                           // offset: 0
// [[no_unique_address]] __empty __outer_it;     //         0
// variant<_PatternIter, _InnerIter> __pattern_; //         sizeof(pointer)
static_assert(sizeof(std::ranges::iterator_t<JWV>) ==
              sizeof(void*) + sizeof(std::variant<int*, int*>)); // sizeof(__parent_) + sizeof(__inner_it_)
