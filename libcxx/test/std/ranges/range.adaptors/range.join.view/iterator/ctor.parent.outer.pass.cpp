//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// constexpr iterator(Parent& parent, OuterIter outer);

#include <cassert>
#include <ranges>

#include "../types.h"

using NonDefaultCtrIter = cpp20_input_iterator<int*>;
static_assert(!std::default_initializable<NonDefaultCtrIter>);

using NonDefaultCtrIterView = BufferView<NonDefaultCtrIter, sentinel_wrapper<NonDefaultCtrIter>>;
static_assert(std::ranges::input_range<NonDefaultCtrIterView>);

constexpr bool test() {
  int buffer[4][4] = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}};
  {
    CopyableChild children[4] = {CopyableChild(buffer[0]), CopyableChild(buffer[1]), CopyableChild(buffer[2]),
                                 CopyableChild(buffer[3])};
    CopyableParent parent{children};
    std::ranges::join_view jv(parent);
    std::ranges::iterator_t<decltype(jv)> iter(jv, std::ranges::begin(parent));
    assert(*iter == 1);
  }

  {
    // LWG3569 Inner iterator not default_initializable
    // With the current spec, the constructor under test invokes Inner iterator's default constructor
    // even if it is not default constructible
    // This test is checking that this constructor can be invoked with an inner range with non default
    // constructible iterator
    NonDefaultCtrIterView inners[] = {buffer[0], buffer[1]};
    auto outer = std::views::all(inners);
    std::ranges::join_view jv(outer);
    std::ranges::iterator_t<decltype(jv)> iter(jv, std::ranges::begin(outer));
    assert(*iter == 1);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
