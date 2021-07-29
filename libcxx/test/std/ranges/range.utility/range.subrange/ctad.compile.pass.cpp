//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: gcc-10
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// class std::ranges::subrange;

#include <ranges>

#include <cassert>
#include "test_macros.h"
#include "test_iterators.h"

using FI = forward_iterator<int*>;
FI fi{nullptr};
int *ptr = nullptr;

static_assert(std::same_as<decltype(std::ranges::subrange(fi, fi)),
                           std::ranges::subrange<FI, FI, std::ranges::subrange_kind::unsized>>);
static_assert(std::same_as<decltype(std::ranges::subrange(ptr, ptr, 0)),
                           std::ranges::subrange<int*, int*, std::ranges::subrange_kind::sized>>);
static_assert(std::same_as<decltype(std::ranges::subrange(ptr, nullptr, 0)),
                           std::ranges::subrange<int*, nullptr_t, std::ranges::subrange_kind::sized>>);

struct ForwardRange {
  forward_iterator<int*> begin() const;
  forward_iterator<int*> end() const;
};
template<>
inline constexpr bool std::ranges::enable_borrowed_range<ForwardRange> = true;

struct SizedRange {
  int *begin();
  int *end();
};
template<>
inline constexpr bool std::ranges::enable_borrowed_range<SizedRange> = true;

static_assert(std::same_as<decltype(std::ranges::subrange(ForwardRange())),
                           std::ranges::subrange<FI, FI, std::ranges::subrange_kind::unsized>>);
static_assert(std::same_as<decltype(std::ranges::subrange(SizedRange())),
                           std::ranges::subrange<int*, int*, std::ranges::subrange_kind::sized>>);
static_assert(std::same_as<decltype(std::ranges::subrange(SizedRange(), 8)),
                           std::ranges::subrange<int*, int*, std::ranges::subrange_kind::sized>>);
