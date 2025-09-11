//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <ranges>

// constexpr iterator& operator++();
// constexpr void operator++(int);
// constexpr iterator operator++(int)
//   requires ref-is-glvalue && forward_iterator<OuterIter> &&
//            forward_iterator<InnerIter>;

#include <ranges>

#include <array>
#include <cassert>
#include <type_traits>
#include <vector>

#include "../types.h"

template <class I>
concept CanPreIncrement = requires(I& i) { ++i; };

template <class I>
concept CanPostIncrement = requires(I& i) { i++; };

template <bool RefIsGlvalue, class Inner>
using VRange = std::conditional_t<RefIsGlvalue, std::vector<Inner>, RvalueVector<Inner>>;

template <bool RefIsGlvalue>
constexpr void test_pre_increment() {
  { // `V` and `Pattern` are not empty. Test return type too.
    using V       = VRange<RefIsGlvalue, std::array<int, 2>>;
    using Pattern = std::array<int, 2>;
    using JWV     = std::ranges::join_with_view<std::ranges::owning_view<V>, std::ranges::owning_view<Pattern>>;

    JWV jwv(V{{1, 1}, {2, 2}, {3, 3}}, Pattern{0, 0});

    {
      using Iter = std::ranges::iterator_t<JWV>;
      static_assert(CanPreIncrement<Iter>);
      static_assert(!CanPreIncrement<const Iter>);

      auto it = jwv.begin();
      assert(*it == 1);
      std::same_as<Iter&> decltype(auto) it_ref = ++it;
      if constexpr (RefIsGlvalue) {
        assert(it_ref == it);
      }

      ++it;
      assert(*it == 0);
      ++it_ref;
      ++it_ref;
      assert(*it_ref == 2);
      ++it;
      ++it_ref;
      assert(*it == 0);
    }

    if constexpr (RefIsGlvalue) {
      using CIter = std::ranges::iterator_t<const JWV>;
      static_assert(CanPreIncrement<CIter>);
      static_assert(!CanPreIncrement<const CIter>);

      auto cit = std::as_const(jwv).begin();
      assert(*cit == 1);
      std::same_as<CIter&> decltype(auto) cit_ref = ++cit;
      assert(cit_ref == cit);
      ++cit;
      assert(*cit == 0);
      ++cit_ref;
      ++cit_ref;
      assert(*cit_ref == 2);
      ++cit;
      ++cit_ref;
      assert(*cit == 0);
    }
  }

  { // `V` and `Pattern` are empty.
    using V       = VRange<RefIsGlvalue, std::ranges::empty_view<int>>;
    using Pattern = std::ranges::empty_view<int>;
    using JWV     = std::ranges::join_with_view<std::ranges::owning_view<V>, std::ranges::owning_view<Pattern>>;

    JWV jwv = {};

    {
      auto it = jwv.begin();
      assert(it == jwv.end());
    }

    if constexpr (RefIsGlvalue) {
      auto cit = std::as_const(jwv).begin();
      assert(cit == std::as_const(jwv).end());
    }
  }

  { // `Pattern` is empty, `V` is not.
    using V       = VRange<RefIsGlvalue, std::vector<int>>;
    using Pattern = std::vector<int>;
    using JWV     = std::ranges::join_with_view<std::ranges::owning_view<V>, std::ranges::owning_view<Pattern>>;

    JWV jwv(V{{{-1}, {-2}, {-3}}}, Pattern{});

    {
      auto it = jwv.begin();
      assert(*it == -1);
      ++it;
      assert(*it == -2);
      ++it;
      assert(*it == -3);
      ++it;
      assert(it == jwv.end());
    }

    if constexpr (RefIsGlvalue) {
      auto cit = std::as_const(jwv).begin();
      assert(*cit == -1);
      ++cit;
      assert(*cit == -2);
      ++cit;
      assert(*cit == -3);
      ++cit;
      assert(cit == std::as_const(jwv).end());
    }
  }

  { // `V` has empty subrange in the middle, `Pattern` is not empty.
    using V       = VRange<RefIsGlvalue, std::vector<int>>;
    using Pattern = std::ranges::single_view<int>;
    using JWV     = std::ranges::join_with_view<std::ranges::owning_view<V>, Pattern>;

    JWV jwv(V{{1}, {}, {3}}, Pattern{0});

    {
      auto it = jwv.begin();
      assert(*it == 1);
      ++it;
      assert(*it == 0);
      ++it;
      assert(*it == 0);
      ++it;
      assert(*it == 3);
    }

    if constexpr (RefIsGlvalue) {
      auto cit = std::as_const(jwv).begin();
      assert(*cit == 1);
      ++cit;
      assert(*cit == 0);
      ++cit;
      assert(*cit == 0);
      ++cit;
      assert(*cit == 3);
    }
  }

  { // Only last element of `V` is not empty. `Pattern` is not empty.
    using V       = VRange<RefIsGlvalue, std::vector<int>>;
    using Pattern = std::ranges::single_view<int>;
    using JWV     = std::ranges::join_with_view<std::ranges::owning_view<V>, Pattern>;

    JWV jwv(V{{}, {}, {555}}, Pattern{1});

    {
      auto it = jwv.begin();
      assert(*it == 1);
      ++it;
      assert(*it == 1);
      ++it;
      assert(*it == 555);
      ++it;
      assert(it == jwv.end());
    }

    if constexpr (RefIsGlvalue) {
      auto cit = std::as_const(jwv).begin();
      assert(*cit == 1);
      ++cit;
      assert(*cit == 1);
      ++cit;
      assert(*cit == 555);
      ++cit;
      assert(cit == std::as_const(jwv).end());
    }
  }

  { // Only first element of `V` is not empty. `Pattern` is empty.
    using V       = VRange<RefIsGlvalue, std::vector<int>>;
    using Pattern = std::ranges::empty_view<int>;
    using JWV     = std::ranges::join_with_view<std::ranges::owning_view<V>, Pattern>;

    JWV jwv(V{{777}, {}, {}}, Pattern{});

    {
      auto it = jwv.begin();
      assert(*it == 777);
      ++it;
      assert(it == jwv.end());
    }

    if constexpr (RefIsGlvalue) {
      auto cit = std::as_const(jwv).begin();
      assert(*cit == 777);
      ++cit;
      assert(cit == std::as_const(jwv).end());
    }
  }

  { // Only last element of `V` is not empty. `Pattern` is empty. `V` models input range.
    using V       = BasicView<VRange<RefIsGlvalue, std::string>, ViewProperties{}, DefaultCtorInputIter>;
    using Pattern = std::ranges::empty_view<char>;
    using JWV     = std::ranges::join_with_view<V, Pattern>;

    JWV jwv(V{{}, {}, {'a'}}, Pattern{});

    auto it = jwv.begin();
    assert(*it == 'a');
    ++it;
    assert(it == jwv.end());
  }

  { // Only first element of `V` is not empty. `Pattern` is not empty. `V` models input range.
    using V       = BasicView<VRange<RefIsGlvalue, std::string>, ViewProperties{}, DefaultCtorInputIter>;
    using Pattern = std::ranges::single_view<char>;
    using JWV     = std::ranges::join_with_view<V, Pattern>;

    JWV jwv(V{{'b'}, {}, {}}, Pattern{'.'});

    auto it = jwv.begin();
    assert(*it == 'b');
    ++it;
    assert(*it == '.');
    ++it;
    assert(*it == '.');
    ++it;
    assert(it == jwv.end());
  }
}

constexpr void test_post_increment() {
  { // `V` and `Pattern` are not empty. Return type should be `iterator`.
    using V       = std::array<std::array<int, 3>, 2>;
    using Pattern = std::array<int, 1>;
    using JWV     = std::ranges::join_with_view<std::ranges::owning_view<V>, std::ranges::owning_view<Pattern>>;

    using Iter  = std::ranges::iterator_t<JWV>;
    using CIter = std::ranges::iterator_t<const JWV>;
    static_assert(CanPostIncrement<Iter>);
    static_assert(!CanPostIncrement<const Iter>);
    static_assert(CanPostIncrement<CIter>);
    static_assert(!CanPostIncrement<const CIter>);

    JWV jwv(V{{{6, 5, 4}, {3, 2, 1}}}, Pattern{-5});

    {
      auto it = jwv.begin();
      assert(*it == 6);
      std::same_as<Iter> decltype(auto) it_copy = it++;
      assert(++it_copy == it);
      it++;
      it++;
      assert(*it == -5);
      it_copy++;
      it_copy++;
      assert(*it_copy == -5);
      it++;
      it_copy++;
      assert(*it == 3);
      assert(*it_copy == 3);
    }

    {
      auto cit = std::as_const(jwv).begin();
      assert(*cit == 6);
      std::same_as<CIter> decltype(auto) cit_copy = cit++;
      assert(++cit_copy == cit);
      cit++;
      cit++;
      assert(*cit == -5);
      cit_copy++;
      cit_copy++;
      assert(*cit_copy == -5);
      cit++;
      cit_copy++;
      assert(*cit == 3);
      assert(*cit_copy == 3);
    }
  }

  { // `Pattern` is empty, `V` is not. Value of `ref-is-glvalue` is false (return type should be `void`).
    using Inner   = std::vector<int>;
    using V       = RvalueVector<Inner>;
    using Pattern = std::ranges::empty_view<int>;
    using JWV     = std::ranges::join_with_view<std::ranges::owning_view<V>, std::ranges::owning_view<Pattern>>;

    JWV jwv(V{Inner{-3}, Inner{-2}, Inner{-1}}, Pattern{});

    auto it = jwv.begin();
    assert(*it == -3);
    it++;
    assert(*it == -2);
    it++;
    assert(*it == -1);
    it++;
    assert(it == jwv.end());
    static_assert(std::is_void_v<decltype(it++)>);
  }

  { // `V` has empty subrange in the middle, `Pattern` is not empty.
    // OuterIter does not model forward iterator (return type should be `void`).
    using Inner   = std::vector<int>;
    using V       = BasicVectorView<Inner, ViewProperties{.common = false}, cpp20_input_iterator>;
    using Pattern = std::ranges::single_view<int>;
    using JWV     = std::ranges::join_with_view<V, Pattern>;

    JWV jwv(V{Inner{7}, {}, Inner{9}}, Pattern{8});

    auto it = jwv.begin();
    assert(*it == 7);
    it++;
    assert(*it == 8);
    it++;
    assert(*it == 8);
    it++;
    assert(*it == 9);
    it++;
    assert(it == jwv.end());
    static_assert(std::is_void_v<decltype(it++)>);
  }

#if !defined(TEST_COMPILER_GCC) // GCC c++/101777
  { // Only first element of `V` is not empty. `Pattern` is empty. InnerIter does not model forward
    // iterator (return type should be `void`).
    using Inner   = BasicVectorView<char32_t, ViewProperties{.common = false}, cpp17_input_iterator>;
    using V       = std::array<Inner, 3>;
    using Pattern = std::ranges::empty_view<char32_t>;
    using JWV     = std::ranges::join_with_view<std::ranges::owning_view<V>, Pattern>;

    JWV jwv(V{Inner{U'?'}, Inner{}, Inner{}}, Pattern{});

    auto it = jwv.begin();
    assert(*it == U'?');
    it++;
    assert(it == jwv.end());
    static_assert(std::is_void_v<decltype(it++)>);
  }
#endif // !defined(TEST_COMPILER_GCC)
}

constexpr bool test() {
  test_pre_increment<false>();
  test_pre_increment<true>();
  test_post_increment();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
