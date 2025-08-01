//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// XFAIL: msvc

// <ranges>

// This test ensures that we use `[[no_unique_address]]` in `join_with_view::sentinel`.

#include <cstddef>
#include <ranges>
#include <string_view>

template <bool Const>
struct Iter {
  using value_type      = std::string_view;
  using difference_type = std::ptrdiff_t;

  Iter& operator++();
  Iter operator++(int);
  value_type& operator*() const;
  bool operator==(const Iter&) const;
  bool operator==(std::default_sentinel_t) const;
};

struct View : std::ranges::view_base {
  Iter<false> begin();
  Iter<true> begin() const;
  std::default_sentinel_t end() const;
};

using JWV = std::ranges::join_with_view<View, std::string_view>;

template <class View>
struct Test {
  [[no_unique_address]] std::ranges::sentinel_t<View> se;
  unsigned char pad;
};

static_assert(sizeof(Test<JWV>) == 1);
static_assert(sizeof(Test<const JWV>) == 1);
