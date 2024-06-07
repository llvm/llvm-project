//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <iterator>

#include "test_iterators.h"

using sized_it = random_access_iterator<int*>;
static_assert(std::sized_sentinel_for<sized_it, sized_it>);
static_assert(std::sized_sentinel_for<std::move_iterator<sized_it>, std::move_iterator<sized_it>>);

struct unsized_it {
  using value_type      = int;
  using difference_type = std::ptrdiff_t;

  value_type& operator*() const;
  unsized_it& operator++();
  bool operator==(const unsized_it&) const;
  difference_type operator-(const unsized_it&) const { return 0; }
};

template <>
inline constexpr bool std::disable_sized_sentinel_for<unsized_it, unsized_it> = true;

static_assert(!std::sized_sentinel_for<unsized_it, unsized_it>);
static_assert(!std::sized_sentinel_for<std::move_iterator<unsized_it>, std::move_iterator<unsized_it>>);
