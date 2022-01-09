//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// constexpr auto size() requires sized_range<R>
// constexpr auto size() const requires sized_range<const R>

#include <ranges>

#include <array>
#include <cassert>
#include <concepts>

#include "test_iterators.h"
#include "test_macros.h"

template <class T>
concept HasSize = requires (T t) {
  t.size();
};

constexpr bool test()
{
  {
    struct SubtractableIters {
      forward_iterator<int*> begin();
      sized_sentinel<forward_iterator<int*>> end();
    };
    using OwningView = std::ranges::owning_view<SubtractableIters>;
    static_assert(std::ranges::sized_range<OwningView&>);
    static_assert(!std::ranges::range<const OwningView&>); // no begin/end
    static_assert(HasSize<OwningView&>);
    static_assert(HasSize<OwningView&&>);
    static_assert(!HasSize<const OwningView&>);
    static_assert(!HasSize<const OwningView&&>);
  }
  {
    struct NoSize {
      bidirectional_iterator<int*> begin();
      bidirectional_iterator<int*> end();
    };
    using OwningView = std::ranges::owning_view<NoSize>;
    static_assert(!HasSize<OwningView&>);
    static_assert(!HasSize<OwningView&&>);
    static_assert(!HasSize<const OwningView&>);
    static_assert(!HasSize<const OwningView&&>);
  }
  {
    struct SizeMember {
      bidirectional_iterator<int*> begin();
      bidirectional_iterator<int*> end();
      int size() const;
    };
    using OwningView = std::ranges::owning_view<SizeMember>;
    static_assert(std::ranges::sized_range<OwningView&>);
    static_assert(!std::ranges::range<const OwningView&>); // no begin/end
    static_assert(HasSize<OwningView&>);
    static_assert(HasSize<OwningView&&>);
    static_assert(!HasSize<const OwningView&>); // not a range, therefore no size()
    static_assert(!HasSize<const OwningView&&>);
  }
  {
    // Test an empty view.
    int a[] = {1};
    auto ov = std::ranges::owning_view(std::ranges::subrange(a, a));
    assert(ov.size() == 0);
    assert(std::as_const(ov).size() == 0);
  }
  {
    // Test a non-empty view.
    int a[] = {1};
    auto ov = std::ranges::owning_view(std::ranges::subrange(a, a+1));
    assert(ov.size() == 1);
    assert(std::as_const(ov).size() == 1);
  }
  {
    // Test a non-view.
    std::array<int, 2> a = {1, 2};
    auto ov = std::ranges::owning_view(std::move(a));
    assert(ov.size() == 2);
    assert(std::as_const(ov).size() == 2);
  }
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
