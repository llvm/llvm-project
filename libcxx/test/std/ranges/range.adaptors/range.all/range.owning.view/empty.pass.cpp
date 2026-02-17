//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// constexpr bool empty() requires requires { ranges::empty(r_); }
// constexpr bool empty() const requires requires { ranges::empty(r_); }

#include <ranges>

#include <array>
#include <cassert>
#include <concepts>

#include "test_iterators.h"
#include "test_macros.h"

template <class T>
concept HasEmpty = requires (T t) {
  t.empty();
};

constexpr bool test()
{
  {
    struct ComparableIters {
      forward_iterator<int*> begin();
      forward_iterator<int*> end();
    };
    using OwningView = std::ranges::owning_view<ComparableIters>;
    static_assert(HasEmpty<OwningView&>);
    static_assert(HasEmpty<OwningView&&>);
    static_assert(!HasEmpty<const OwningView&>);
    static_assert(!HasEmpty<const OwningView&&>);
  }
  {
    struct NoEmpty {
      cpp20_input_iterator<int*> begin();
      sentinel_wrapper<cpp20_input_iterator<int*>> end();
    };
    static_assert(std::ranges::range<NoEmpty&>);
    static_assert(!std::invocable<decltype(std::ranges::empty), NoEmpty&>);
    static_assert(!std::ranges::range<const NoEmpty&>); // no begin/end
    static_assert(!std::invocable<decltype(std::ranges::empty), const NoEmpty&>);
    using OwningView = std::ranges::owning_view<NoEmpty>;
    static_assert(!HasEmpty<OwningView&>);
    static_assert(!HasEmpty<OwningView&&>);
    static_assert(!HasEmpty<const OwningView&>);
    static_assert(!HasEmpty<const OwningView&&>);
  }
  {
    struct EmptyMember {
      cpp20_input_iterator<int*> begin();
      sentinel_wrapper<cpp20_input_iterator<int*>> end();
      bool empty() const;
    };
    static_assert(std::ranges::range<EmptyMember&>);
    static_assert(std::invocable<decltype(std::ranges::empty), EmptyMember&>);
    static_assert(!std::ranges::range<const EmptyMember&>); // no begin/end
    static_assert(std::invocable<decltype(std::ranges::empty), const EmptyMember&>);
    using OwningView = std::ranges::owning_view<EmptyMember>;
    static_assert(std::ranges::range<OwningView&>);
    static_assert(!std::ranges::range<const OwningView&>); // no begin/end
    static_assert(HasEmpty<OwningView&>);
    static_assert(HasEmpty<OwningView&&>);
    static_assert(HasEmpty<const OwningView&>); // but it still has empty()
    static_assert(HasEmpty<const OwningView&&>);
  }
  {
    // Test an empty view.
    int a[] = {1};
    auto ov = std::ranges::owning_view(std::ranges::subrange(a, a));
    assert(ov.empty());
    assert(std::as_const(ov).empty());
  }
  {
    // Test a non-empty view.
    int a[] = {1};
    auto ov = std::ranges::owning_view(std::ranges::subrange(a, a+1));
    assert(!ov.empty());
    assert(!std::as_const(ov).empty());
  }
  {
    // Test a non-view.
    std::array<int, 2> a = {1, 2};
    auto ov = std::ranges::owning_view(std::move(a));
    assert(!ov.empty());
    assert(!std::as_const(ov).empty());
  }
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
