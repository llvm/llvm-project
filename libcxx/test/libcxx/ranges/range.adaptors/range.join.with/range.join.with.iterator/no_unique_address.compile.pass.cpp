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

#include <ranges>

#include "test_iterators.h"
#include "test_range.h"

struct InputView : std::ranges::view_base {
  using Inner = test_range<forward_iterator>;

  cpp20_input_iterator<Inner*> begin();
  sentinel_wrapper<cpp20_input_iterator<Inner*>> end();
};

static_assert(std::ranges::input_range<InputView>);
static_assert(!std::ranges::forward_range<InputView>);

struct Pattern : std::ranges::view_base {
  int* begin();
  int* end();
};

static_assert(alignof(void*) == alignof(std::variant<int*, int*>)); // alignof(__parent_) == alignof(__inner_it_)
static_assert(sizeof(std::ranges::iterator_t<std::ranges::join_with_view<InputView, Pattern>>) ==
              sizeof(void*) + sizeof(std::variant<int*, int*>)); // sizeof(__parent_) + sizeof(__inner_it_)
