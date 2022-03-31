//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// constexpr auto end();
// constexpr auto end() const requires range<const V>;

#include <ranges>

#include <cassert>
#include <concepts>
#include <utility>

#include "test_macros.h"
#include "test_iterators.h"
#include "types.h"

constexpr bool test() {
  int buf[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    SizedRandomAccessView view{buf, buf + 8};
    std::ranges::common_view<SizedRandomAccessView> common(view);
    std::same_as<RandomAccessIter> auto end = common.end(); // Note this should NOT be the sentinel type.
    assert(base(end) == buf + 8);
  }

  // const version
  {
    SizedRandomAccessView view{buf, buf + 8};
    std::ranges::common_view<SizedRandomAccessView> const common(view);
    std::same_as<RandomAccessIter> auto end = common.end(); // Note this should NOT be the sentinel type.
    assert(base(end) == buf + 8);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  {
    int buf[8] = {1, 2, 3, 4, 5, 6, 7, 8};

    using CommonForwardIter = std::common_iterator<ForwardIter, sized_sentinel<ForwardIter>>;
    using CommonIntIter = std::common_iterator<int*, sentinel_wrapper<int*>>;

    {
      SizedForwardView view{buf, buf + 8};
      std::ranges::common_view<SizedForwardView> common(view);
      std::same_as<CommonForwardIter> auto end = common.end();
      assert(end == CommonForwardIter(std::ranges::end(view)));
    }
    {
      CopyableView view{buf, buf + 8};
      std::ranges::common_view<CopyableView> common(view);
      std::same_as<CommonIntIter> auto end = common.end();
      assert(end == CommonIntIter(std::ranges::end(view)));
    }

    // const versions
    {
      SizedForwardView view{buf, buf + 8};
      std::ranges::common_view<SizedForwardView> const common(view);
      std::same_as<CommonForwardIter> auto end = common.end();
      assert(end == CommonForwardIter(std::ranges::end(view)));
    }
    {
      CopyableView view{buf, buf + 8};
      std::ranges::common_view<CopyableView> const common(view);
      std::same_as<CommonIntIter> auto end = common.end();
      assert(end == CommonIntIter(std::ranges::end(view)));
    }
  }

  return 0;
}
