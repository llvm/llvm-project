//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// XFAIL: msvc

// This test ensures that we use `[[no_unique_address]]` in `zip_transform_view`.

#include <ranges>

struct View : std::ranges::view_base {
  int* begin() const;
  int* end() const;
};

struct Pred {
  template <class... Args>
  bool operator()(const Args&...) const;
};

template <class View>
struct Test {
  [[no_unique_address]] View view;
  char c;
};

static_assert(sizeof(std::ranges::zip_transform_view<Pred, View>) == 1);
