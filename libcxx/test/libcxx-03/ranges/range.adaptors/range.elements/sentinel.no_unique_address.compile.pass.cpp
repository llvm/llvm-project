//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// clang-cl and cl currently don't support [[no_unique_address]]
// XFAIL: msvc

// Test the libc++ extension that the sentinel stored in `std::ranges::elements_view::__sentinel`
// has been marked as _LIBCPP_NO_UNIQUE_ADDRESS

#include <ranges>
#include <tuple>

struct EmptySentinel {
  friend bool operator==(std::tuple<int>* iter, EmptySentinel) { return iter; }
};

struct Range : std::ranges::view_base {
  std::tuple<int>* begin() const;
  EmptySentinel end() const;
};

using ElementsView = std::ranges::elements_view<Range, 0>;
using ElementsSent = std::ranges::sentinel_t<ElementsView>;

struct TestClass {
  [[no_unique_address]] ElementsSent s;
  int i;
};

static_assert(sizeof(TestClass) == sizeof(int));
