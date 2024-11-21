//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// XFAIL: msvc && clang-17

// <ranges>

// This test ensures that we use `[[no_unique_address]]` in `join_with_view::iterator`.

#include <iterator>
#include <ranges>

struct IntRange : std::ranges::view_base {
  int* begin();
  int* end();
};

class InputIter {
public:
  using value_type      = IntRange;
  using difference_type = ptrdiff_t;

  InputIter& operator++();
  void operator++(int);
  value_type& operator*() const;
  bool operator==(const InputIter&) const;

private:
  int* ptr_;
};

static_assert(std::input_iterator<InputIter>);
static_assert(!std::forward_iterator<InputIter>);

struct InputView : std::ranges::view_base {
  InputIter begin();
  InputIter end();
};

static_assert(std::ranges::input_range<InputView>);
static_assert(!std::ranges::forward_range<InputView>);

static_assert(alignof(void*) == alignof(std::variant<int*, int*>)); // alignof(__parent_) == alignof(__inner_it_)
static_assert(sizeof(std::ranges::iterator_t<std::ranges::join_with_view<InputView, IntRange>>) ==
              sizeof(void*) + sizeof(std::variant<int*, int*>)); // sizeof(__parent_) + sizeof(__inner_it_)
