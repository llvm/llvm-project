//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_ZIP_TRANSFORM_TYPES_H
#define TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_ZIP_TRANSFORM_TYPES_H

#include <functional>
#include <ranges>

#include "test_macros.h"
#include "test_iterators.h"
#include "test_range.h"
#include "../range_adaptor_types.h"

#if TEST_STD_VER <= 20
#  error "range.zip.transform/types.h" can only be included in builds supporting C++20
#endif

struct IntView : std::ranges::view_base {
  int* begin() const;
  int* end() const;
};

struct MakeTuple {
  constexpr auto operator()(auto&&... args) const { return std::tuple(std::forward<decltype(args)>(args)...); }
};

struct Tie {
  constexpr auto operator()(auto&&... args) const { return std::tie(std::forward<decltype(args)>(args)...); }
};

struct GetFirst {
  constexpr decltype(auto) operator()(auto&& first, auto&&...) const { return std::forward<decltype(first)>(first); }
};

struct NoConstBeginView : std::ranges::view_base {
  int* begin();
  int* end();
};

struct ConstNonConstDifferentView : std::ranges::view_base {
  int* begin();
  const int* begin() const;
  int* end();
  const int* end() const;
};

struct NonConstOnlyFn {
  int operator()(int&) const;
  int operator()(const int&) const = delete;
};

#endif // TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_ZIP_TRANSFORM_TYPES_H
