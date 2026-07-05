//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// XFAIL: msvc

// This test ensures that we use `[[no_unique_address]]` in `chunk_by_view`.

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

// [[no_unique_address]] applied to _View
struct ViewWithPadding : View {
  alignas(128) char c;
};

static_assert(sizeof(Test<std::ranges::chunk_by_view<ViewWithPadding, Pred>>) ==
              sizeof(std::ranges::chunk_by_view<ViewWithPadding, Pred>));

// [[no_unique_address]] applied to movable-box
struct PredWithPadding : Pred {
  alignas(128) char c;
};

static_assert(sizeof(Test<std::ranges::chunk_by_view<View, PredWithPadding>>) ==
              sizeof(std::ranges::chunk_by_view<View, PredWithPadding>));
