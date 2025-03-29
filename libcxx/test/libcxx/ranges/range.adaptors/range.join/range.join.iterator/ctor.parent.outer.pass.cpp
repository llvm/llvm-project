//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// constexpr iterator(Parent& parent, OuterIter outer)
//   requires forward_range<Base>; // exposition only

#include <cassert>
#include <ranges>
#include <string>
#include <utility>

#include "types.h"

constexpr bool test() {
  std::string strings[4] = {"aaaa", "bbbb", "cccc", "dddd"};

  { // Check if `outer_` is initialized with `std::move(outer)` for `iterator<false>`
    MoveOnAccessSubrange r{DieOnCopyIterator(strings), sentinel_wrapper(strings + 4)};
    std::ranges::join_view jv(std::move(r));
    auto iter = jv.begin(); // Calls `iterator(Parent& parent, OuterIter outer)`
    assert(*iter == 'a');
  }

  { // Check if `outer_` is initialized with `std::move(outer)` for `iterator<true>`
    MoveOnAccessSubrange r{DieOnCopyIterator(strings), sentinel_wrapper(strings + 4)};
    std::ranges::join_view jv(std::ranges::ref_view{r});
    auto iter = std::as_const(jv).begin(); // Calls `iterator(Parent& parent, OuterIter outer)`
    assert(*iter == 'a');
  }

  {
    // LWG3569 Inner iterator not default_initializable
    // With the current spec, the constructor under test invokes Inner iterator's default constructor
    // even if it is not default constructible.
    // This test is checking that this constructor can be invoked with an inner range with non default
    // constructible iterator.
    using NonDefaultCtrIter = cpp20_input_iterator<int*>;
    static_assert(!std::default_initializable<NonDefaultCtrIter>);
    using NonDefaultCtrIterView = BufferView<NonDefaultCtrIter, sentinel_wrapper<NonDefaultCtrIter>>;
    static_assert(std::ranges::input_range<NonDefaultCtrIterView>);

    int buffer[2][2]               = {{1, 2}, {3, 4}};
    NonDefaultCtrIterView inners[] = {buffer[0], buffer[1]};
    auto outer                     = std::views::all(inners);
    std::ranges::join_view jv(outer);
    auto iter = jv.begin(); // Calls `iterator(Parent& parent, OuterIter outer)`
    assert(*iter == 1);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
