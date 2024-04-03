//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// Test that views that use __movable_box do not overwrite overlapping subobjects.
// https://github.com/llvm/llvm-project/issues/70506

#include <cassert>
#include <ranges>

#include "test_macros.h"

struct Pred {
  alignas(128) bool a{};

  Pred() noexcept            = default;
  Pred(const Pred&) noexcept = default;
  Pred(Pred&&) noexcept      = default;

  Pred& operator=(const Pred&) = delete;
  Pred& operator=(Pred&&)      = delete;

  constexpr bool operator()(const auto&...) const { return true; }
};

struct View : std::ranges::view_base {
  constexpr int* begin() const { return nullptr; }
  constexpr int* end() const { return nullptr; }
};

template <class View>
struct S {
  [[no_unique_address]] View view{};
  char c = 42;
};

template <class View>
constexpr void testOne() {
  S<View> s1;
  assert(s1.c == 42);
  s1.view = View{};
  assert(s1.c == 42);
}

constexpr bool test() {
  testOne<std::ranges::transform_view<View, Pred>>();
  testOne<std::ranges::filter_view<View, Pred>>();
  testOne<std::ranges::drop_while_view<View, Pred>>();
  testOne<std::ranges::take_while_view<View, Pred>>();
  testOne<std::ranges::single_view<Pred>>();

#if TEST_STD_VER >= 23
  testOne<std::ranges::chunk_by_view<View, Pred>>();
  testOne<std::ranges::repeat_view<Pred>>();
#endif
  return true;
}

int main(int, char**) {
  static_assert(test());
  test();

  return 0;
}
