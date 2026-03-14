//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// class std::ranges::lazy_split_view;

#include <ranges>

#include <cassert>
#include <concepts>
#include <string_view>
#include <type_traits>
#include "types.h"

using V = SplitViewForward;

static_assert(std::is_base_of_v<std::ranges::view_interface<SplitViewForward>, SplitViewForward>);

constexpr bool test() {
  using namespace std::string_view_literals;

  // empty()
  {
    {
      std::ranges::lazy_split_view v("abc def", " ");
      assert(!v.empty());
    }

    {
      // Note: an empty string literal would still produce a non-empty output because the terminating zero is treated as
      // a separate character; hence the use of `string_view`.
      std::ranges::lazy_split_view v(""sv, "");
      assert(v.empty());
    }
  }

  // operator bool()
  {
    {
      std::ranges::lazy_split_view v("abc", "");
      assert(v);
    }

    {
      // Note: an empty string literal would still produce a non-empty output because the terminating zero is treated as
      // a separate character; hence the use of `string_view`.
      std::ranges::lazy_split_view v(""sv, "");
      assert(!v);
    }
  }

  // front()
  {
    SplitViewForward v("abc", "");
    assert(*(v.front()).begin() == 'a');
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
