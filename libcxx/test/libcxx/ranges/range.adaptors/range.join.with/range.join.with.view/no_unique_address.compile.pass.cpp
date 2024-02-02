//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// XFAIL: msvc

// <ranges>

// This test ensures that we use `[[no_unique_address]]` in `join_with_view`.

#include <ranges>
#include <string>

struct ForwardView : std::ranges::view_base {
  std::string* begin() const;
  std::string* end() const;
};

static_assert(std::ranges::forward_range<ForwardView>);
static_assert(std::is_reference_v<std::ranges::range_reference_t<ForwardView>>);

struct Pattern : std::ranges::view_base {
  char* begin() const;
  char* end() const;
};

template <class View>
struct Test {
  [[no_unique_address]] View view;
  char c;
};

static_assert(sizeof(Test<std::ranges::join_with_view<ForwardView, Pattern>>) ==
              sizeof(std::ranges::join_with_view<ForwardView, Pattern>));
