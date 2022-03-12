//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// template<class T>
//   inline constexpr bool enable_borrowed_range<common_view<T>> = enable_borrowed_range<T>;

#include <ranges>

// common_view can only wrap Ts that are `view<T> && !common_range<T>`, so we need to invent one.
struct Uncommon : std::ranges::view_base {
    int *begin() const;
    std::unreachable_sentinel_t end() const;
};

static_assert(!std::ranges::borrowed_range<std::ranges::common_view<Uncommon>>);
static_assert(!std::ranges::borrowed_range<std::ranges::common_view<std::ranges::owning_view<Uncommon>>>);
static_assert( std::ranges::borrowed_range<std::ranges::common_view<std::ranges::ref_view<Uncommon>>>);
