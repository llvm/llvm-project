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

// Test the libc++ extension that the base view stored in `std::ranges::elements_view`
// has been marked as _LIBCPP_NO_UNIQUE_ADDRESS

#include <ranges>
#include <tuple>


struct EmptyView : std::ranges::view_base {
  std::tuple<int>* begin() const;
  std::tuple<int>* end() const;
};

using ElementsView = std::ranges::elements_view<EmptyView, 0>;

struct TestClass {
  [[no_unique_address]] ElementsView view;
  int i;
};

static_assert(sizeof(TestClass) == sizeof(int));
