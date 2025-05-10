//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <ranges>

// constexpr iterator& operator--()
//   requires ref-is-glvalue && bidirectional_range<Base> &&
//            bidirectional-common<InnerBase> && bidirectional-common<PatternBase>;
// constexpr iterator operator--(int)
//   requires ref-is-glvalue && bidirectional_range<Base> &&
//            bidirectional-common<InnerBase> && bidirectional-common<PatternBase>;

#include <ranges>

#include <algorithm>
#include <array>
#include <cassert>
#include <forward_list>
#include <string>
#include <string_view>
#include <vector>

#include "../types.h"

template <class I>
concept CanPreDecrement = requires(I& i) {
  { --i } -> std::same_as<I&>;
};

template <class I>
concept CanPostDecrement = requires(I& i) {
  { i-- } -> std::same_as<I>;
};

template <class I>
concept CanDecrement = CanPreDecrement<I> && CanPostDecrement<I>;

constexpr bool test() {
  { // `V` and `Pattern` are not empty. Test return type too.
    using V       = std::ranges::owning_view<std::vector<std::string>>;
    using Pattern = std::ranges::single_view<char>;
    using JWV     = std::ranges::join_with_view<V, Pattern>;

    using Iter  = std::ranges::iterator_t<JWV>;
    using CIter = std::ranges::iterator_t<const JWV>;
    static_assert(CanDecrement<Iter>);
    static_assert(CanDecrement<CIter>);

    JWV jwv(V{{"01", "23", "45"}}, Pattern{'_'});

    {
      auto it                                   = jwv.end();
      std::same_as<Iter&> decltype(auto) it_ref = --it;
      assert(it_ref == it);
      assert(*it == '5');
      std::same_as<Iter> decltype(auto) it_copy = it--;
      assert(--it_copy == it);
      --it;
      assert(*it == '_');
      it--;
      assert(*it == '3');
      --it;
      it--;
      assert(*it == '_');
    }

    {
      auto cit                                    = std::as_const(jwv).end();
      std::same_as<CIter&> decltype(auto) cit_ref = --cit;
      assert(cit_ref == cit);
      assert(*cit == '5');
      std::same_as<CIter> decltype(auto) cit_copy = cit--;
      assert(--cit_copy == cit);
      --cit;
      assert(*cit == '_');
      cit--;
      assert(*cit == '3');
      --cit;
      cit--;
      assert(*cit == '_');
    }

    assert(std::ranges::equal(std::views::reverse(std::move(jwv)), std::string_view{"54_32_10"}));
  }

  { // `Pattern` is empty, `V` is not.
    using Inner   = std::array<int, 1>;
    using V       = std::ranges::owning_view<std::array<Inner, 3>>;
    using Pattern = std::ranges::owning_view<std::array<int, 0>>;
    using JWV     = std::ranges::join_with_view<V, Pattern>;

    JWV jwv(V{{Inner{-9}, Inner{-99}, Inner{-999}}}, Pattern{});

    {
      auto it = jwv.end();
      --it;
      assert(*it == -999);
      it--;
      assert(*it == -99);
      --it;
      assert(*it == -9);
      assert(it == jwv.begin());
    }

    {
      auto cit = std::as_const(jwv).end();
      --cit;
      assert(*cit == -999);
      cit--;
      assert(*cit == -99);
      --cit;
      assert(*cit == -9);
      assert(cit == std::as_const(jwv).begin());
    }
  }

#if !defined(TEST_COMPILER_GCC) // GCC c++/101777
  { // `V` has empty subrange in the middle, `Pattern` is not empty. Try to go back and forth.
    using V       = std::array<std::vector<int>, 3>;
    using Pattern = std::ranges::single_view<int>;
    using JWV     = std::ranges::join_with_view<std::ranges::owning_view<V>, Pattern>;

    JWV jwv(V{{{5}, {}, {125}}}, Pattern{1});

    {
      auto it = jwv.end();
      --it;
      assert(*it == 125);
      it--;
      assert(*it == 1);
      --it;
      assert(*it == 1);
      it--;
      assert(*it == 5);
      ++it;
      assert(*it == 1);
      --it;
      assert(*it == 5);
      std::ranges::advance(it, 4);
      it--;
      assert(*it == 125);
    }

    {
      auto cit = std::as_const(jwv).end();
      --cit;
      assert(*cit == 125);
      cit--;
      assert(*cit == 1);
      --cit;
      assert(*cit == 1);
      cit--;
      assert(*cit == 5);
      ++cit;
      assert(*cit == 1);
      --cit;
      assert(*cit == 5);
      std::ranges::advance(cit, 4);
      cit--;
      assert(*cit == 125);
    }
  }

  { // Only first element of `V` is not empty. `Pattern` is empty. Try to go back and forth.
    using Inner   = std::vector<int>;
    using V       = std::ranges::owning_view<std::array<Inner, 3>>;
    using Pattern = std::ranges::empty_view<int>;
    using JWV     = std::ranges::join_with_view<V, Pattern>;

    JWV jwv(V{{Inner{999}, {}, {}}}, Pattern{});

    {
      auto it = jwv.end();
      --it;
      assert(*it == 999);
      ++it;
      assert(it == jwv.end());
      it--;
      assert(*it == 999);
    }

    {
      auto cit = std::as_const(jwv).end();
      --cit;
      assert(*cit == 999);
      ++cit;
      assert(cit == std::as_const(jwv).end());
      cit--;
      assert(*cit == 999);
    }
  }
#endif // !defined(TEST_COMPILER_GCC)

  { // `ref-is-glvalue` is false
    using V       = RvalueVector<std::vector<int>>;
    using Pattern = std::ranges::empty_view<int>;
    using JWV     = std::ranges::join_with_view<std::ranges::owning_view<V>, std::ranges::owning_view<Pattern>>;
    using Iter    = std::ranges::iterator_t<JWV>;
    static_assert(!CanPreDecrement<Iter>);
    static_assert(!CanPostDecrement<Iter>);
  }

  { // `Base` does not model bidirectional range
    using V       = std::ranges::owning_view<std::forward_list<std::vector<int>>>;
    using Pattern = std::ranges::single_view<int>;
    using JWV     = std::ranges::join_with_view<V, Pattern>;
    using Iter    = std::ranges::iterator_t<JWV>;
    using CIter   = std::ranges::iterator_t<const JWV>;
    static_assert(!CanPreDecrement<Iter>);
    static_assert(!CanPostDecrement<Iter>);
    static_assert(!CanPreDecrement<CIter>);
    static_assert(!CanPostDecrement<CIter>);
  }

  {   // InnerBase does not model bidirectional-common
    { // InnerBase does not model bidirectional range
      using V       = std::ranges::owning_view<std::vector<std::forward_list<int>>>;
      using Pattern = std::ranges::single_view<int>;
      using JWV     = std::ranges::join_with_view<V, Pattern>;
      using Iter    = std::ranges::iterator_t<JWV>;
      using CIter   = std::ranges::iterator_t<const JWV>;
      static_assert(!CanPreDecrement<Iter>);
      static_assert(!CanPostDecrement<Iter>);
      static_assert(!CanPreDecrement<CIter>);
      static_assert(!CanPostDecrement<CIter>);
    }

    { // InnerBase does not model common range
      using InnerBase = BasicVectorView<int, ViewProperties{.common = false}, bidirectional_iterator>;
      using V         = std::ranges::owning_view<std::vector<InnerBase>>;
      using Pattern   = std::ranges::single_view<int>;
      using JWV       = std::ranges::join_with_view<V, Pattern>;
      using Iter      = std::ranges::iterator_t<JWV>;
      using CIter     = std::ranges::iterator_t<const JWV>;
      static_assert(!CanPreDecrement<Iter>);
      static_assert(!CanPostDecrement<Iter>);
      static_assert(!CanPreDecrement<CIter>);
      static_assert(!CanPostDecrement<CIter>);
    }
  }

  {   // PatternBase does not model bidirectional-common
    { // PatternBase does not model bidirectional range
      using V       = std::ranges::owning_view<std::vector<std::vector<int>>>;
      using Pattern = std::ranges::owning_view<std::forward_list<int>>;
      using JWV     = std::ranges::join_with_view<V, Pattern>;
      using Iter    = std::ranges::iterator_t<JWV>;
      using CIter   = std::ranges::iterator_t<const JWV>;
      static_assert(!CanPreDecrement<Iter>);
      static_assert(!CanPostDecrement<Iter>);
      static_assert(!CanPreDecrement<CIter>);
      static_assert(!CanPostDecrement<CIter>);
    }

    { // PatternBase does not model common range
      using V       = std::ranges::owning_view<std::vector<std::vector<int>>>;
      using Pattern = BasicVectorView<int, ViewProperties{.common = false}, bidirectional_iterator>;
      using JWV     = std::ranges::join_with_view<V, Pattern>;
      using Iter    = std::ranges::iterator_t<JWV>;
      using CIter   = std::ranges::iterator_t<const JWV>;
      static_assert(!CanPreDecrement<Iter>);
      static_assert(!CanPostDecrement<Iter>);
      static_assert(!CanPreDecrement<CIter>);
      static_assert(!CanPostDecrement<CIter>);
    }
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
