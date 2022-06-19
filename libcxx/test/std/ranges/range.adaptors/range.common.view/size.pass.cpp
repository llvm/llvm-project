//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// constexpr auto size() requires sized_range<V>
// constexpr auto size() const requires sized_range<const V>

#include <ranges>
#include <cassert>

#include "test_iterators.h"
#include "types.h"

template<class View>
concept SizeEnabled = requires(View v) { v.size(); };

constexpr bool test() {
  int buf[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    static_assert( SizeEnabled<std::ranges::common_view<SizedForwardView> const&>);
    static_assert(!SizeEnabled<std::ranges::common_view<CopyableView> const&>);
  }

  {
    SizedForwardView view{buf, buf + 8};
    std::ranges::common_view<SizedForwardView> common(view);
    assert(common.size() == 8);
  }

  {
    SizedForwardView view{buf, buf + 8};
    std::ranges::common_view<SizedForwardView> const common(view);
    assert(common.size() == 8);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
