//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// constexpr const iterator_t<Base>& base() const & noexcept;
// constexpr iterator_t<Base> base() &&;

#include <cassert>
#include <ranges>
#include <tuple>
#include <type_traits>
#include <utility>

#include "MoveOnly.h"
#include "../types.h"

// Test Noexcept
template <class T>
concept IsBaseNoexcept =
    requires {
      { std::declval<T>().base() } noexcept;
    };

using BaseView     = std::ranges::subrange<std::tuple<int>*>;
using ElementsIter = std::ranges::iterator_t<std::ranges::elements_view<BaseView, 0>>;

static_assert(IsBaseNoexcept<const ElementsIter&>);
static_assert(IsBaseNoexcept<ElementsIter&>);
static_assert(IsBaseNoexcept<const ElementsIter&&>);
LIBCPP_STATIC_ASSERT(!IsBaseNoexcept<ElementsIter&&>);

constexpr bool test() {
  std::tuple<int> t{5};

  // const &
  {
    const ElementsIter it{&t};
    decltype(auto) base = it.base();
    static_assert(std::is_same_v<decltype(base), std::tuple<int>* const&>);
    assert(base == &t);
  }

  // &
  {
    ElementsIter it{&t};
    decltype(auto) base = it.base();
    static_assert(std::is_same_v<decltype(base), std::tuple<int>* const&>);
    assert(base == &t);
  }

  // &&
  {
    ElementsIter it{&t};
    decltype(auto) base = std::move(it).base();
    static_assert(std::is_same_v<decltype(base), std::tuple<int>*>);
    assert(base == &t);
  }

  // const &&
  {
    const ElementsIter it{&t};
    decltype(auto) base = std::move(it).base();
    static_assert(std::is_same_v<decltype(base), std::tuple<int>* const&>);
    assert(base == &t);
  }

  // move only
  {
    struct MoveOnlyIter : IterBase<MoveOnlyIter> {
      MoveOnly mo;
    };
    struct Sent {
      constexpr bool operator==(const MoveOnlyIter&) const { return true; }
    };

    using MoveOnlyElemIter =
        std::ranges::iterator_t<std::ranges::elements_view<std::ranges::subrange<MoveOnlyIter, Sent>, 0>>;

    MoveOnlyElemIter it{MoveOnlyIter{{}, MoveOnly{5}}};
    decltype(auto) base = std::move(it).base();
    static_assert(std::is_same_v<decltype(base), MoveOnlyIter>);
    assert(base.mo.get() == 5);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
