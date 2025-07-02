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
  unsigned char pad;
};

using JWV = std::ranges::join_with_view<ForwardView, Pattern>;

// Expected JWV layout:
// [[no_unique_address]] _View __base_             // offset: 0
// [[no_unique_address]] __empty_cache __outer_it; //         0
// [[no_unique_address]] __empty_cache __inner_;   //         1
// [[no_unique_address]] _Patter __pattern_        //         0
static_assert(sizeof(JWV) == 2);
static_assert(sizeof(Test<JWV>) == 2);
