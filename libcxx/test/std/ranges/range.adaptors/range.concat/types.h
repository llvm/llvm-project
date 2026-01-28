//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_CONCAT_TYPES_H
#define TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_CONCAT_TYPES_H

#include <ranges>
#include <utility>
#include "test_iterators.h"

template <class Iter, class Sent>
struct minimal_view : std::ranges::view_base {
  constexpr explicit minimal_view(Iter it, Sent sent) : it_(base(std::move(it))), sent_(base(std::move(sent))) {}

  minimal_view(minimal_view&&)            = default;
  minimal_view& operator=(minimal_view&&) = default;

  constexpr Iter begin() const { return Iter(it_); }
  constexpr Sent end() const { return Sent(sent_); }

private:
  decltype(base(std::declval<Iter>())) it_;
  decltype(base(std::declval<Sent>())) sent_;
};

#endif // TEST_STD_RANGES_RANGE_ADAPTORS_CONCAT_FILTER_TYPES_H
