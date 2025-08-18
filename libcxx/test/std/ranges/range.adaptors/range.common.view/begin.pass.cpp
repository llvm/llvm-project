//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// constexpr auto begin();
// constexpr auto begin() const requires range<const V>;

#include <ranges>

#include <cassert>
#include <concepts>
#include <utility>

#include "test_iterators.h"
#include "types.h"

struct MutableView : std::ranges::view_base {
  int* begin();
  sentinel_wrapper<int*> end();
};

template <class T>
concept HasConstBegin = requires(const T& ct) { ct.begin(); };

template <class T>
concept HasBegin = requires(T& t) { t.begin(); };

template <class T>
concept HasConstAndNonConstBegin = HasConstBegin<T> && requires(T& t, const T& ct) {
  requires !std::same_as<decltype(t.begin()), decltype(ct.begin())>;
};

template <class T>
concept HasOnlyNonConstBegin = HasBegin<T> && !HasConstBegin<T>;

template <class T>
concept HasOnlyConstBegin = HasConstBegin<T> && !HasConstAndNonConstBegin<T>;

constexpr bool test() {
  int buf[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    static_assert( HasBegin<std::ranges::common_view<CopyableView> const&>);
    static_assert( HasBegin<std::ranges::common_view<MutableView>&>);
    static_assert(!HasBegin<std::ranges::common_view<MutableView> const&>);
  }

  {
    SizedRandomAccessView view{buf, buf + 8};
    std::ranges::common_view<SizedRandomAccessView> common(view);
    std::same_as<RandomAccessIter> auto begin = common.begin();
    assert(begin == std::ranges::begin(view));
  }

  {
    SizedRandomAccessView view{buf, buf + 8};
    std::ranges::common_view<SizedRandomAccessView> const common(view);
    std::same_as<RandomAccessIter> auto begin = common.begin();
    assert(begin == std::ranges::begin(view));
  }

  {
    NonSimpleNonCommonView view{buf, buf + 8};
    std::ranges::common_view<NonSimpleNonCommonView> common(view);
    static_assert(!std::ranges::__simple_view<NonSimpleNonCommonView>);
    static_assert(!std::is_same_v<decltype(common.begin()), decltype(std::as_const(common).begin())>);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  // The non-constexpr tests:
  int buf[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    SizedForwardView view{buf, buf + 8};
    std::ranges::common_view<SizedForwardView> common(view);
    using CommonIter = std::common_iterator<ForwardIter, sized_sentinel<ForwardIter>>;
    std::same_as<CommonIter> auto begin = common.begin();
    assert(begin == std::ranges::begin(view));
    std::same_as<CommonIter> auto cbegin = std::as_const(common).begin();
    assert(cbegin == std::ranges::begin(view));
  }

  {
    MoveOnlyView view{buf, buf + 8};
    std::ranges::common_view<MoveOnlyView> common(std::move(view));
    using CommonIter = std::common_iterator<int*, sentinel_wrapper<int*>>;
    std::same_as<CommonIter> auto begin = common.begin();
    assert(begin == std::ranges::begin(view));
  }

  {
    CopyableView view{buf, buf + 8};
    std::ranges::common_view<CopyableView> const common(view);
    using CommonIter = std::common_iterator<int*, sentinel_wrapper<int*>>;
    std::same_as<CommonIter> auto begin = common.begin();
    assert(begin == std::ranges::begin(view));
  }

  return 0;
}
