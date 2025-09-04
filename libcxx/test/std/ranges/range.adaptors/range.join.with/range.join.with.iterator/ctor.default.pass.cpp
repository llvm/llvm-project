//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <ranges>

// iterator() = default;

#include <ranges>

#include <cassert>
#include <initializer_list>
#include <iterator>
#include <type_traits>
#include <utility>

#include "../types.h"
#include "test_comparisons.h"
#include "test_iterators.h"

constexpr bool test() {
  { // `V` and `Pattern` model forward range
    using Inner     = BasicVectorView<int, ViewProperties{}, forward_iterator>;
    using V         = BasicVectorView<Inner, ViewProperties{}, forward_iterator>;
    using Pattern   = Inner;
    using JWV       = std::ranges::join_with_view<V, Pattern>;
    using Iter      = std::ranges::iterator_t<JWV>;
    using ConstIter = std::ranges::iterator_t<const JWV>;

    // Default constructor of iterator<false> should not be explicit
    Iter iter = {};
    assert(testEquality(iter, Iter{}, true));

    // Default constructor of iterator<true> should not be explicit
    ConstIter citer = {};
    assert(testEquality(citer, ConstIter{}, true));
    assert(testEquality(iter, citer, true));

    std::ranges::join_with_view<V, Pattern> jwv(V{Inner{1, 2}, Inner{2, 1}}, Pattern{3, 3});
    Iter jwv_iter       = jwv.begin();
    ConstIter jwv_citer = std::as_const(jwv).begin();
    assert(testEquality(jwv_iter, jwv_citer, true));

    assert(testEquality(jwv_iter, iter, false));
    assert(testEquality(jwv_iter, citer, false));
    assert(testEquality(jwv_citer, iter, false));
    assert(testEquality(jwv_citer, citer, false));
  }

  { // `InnerIter` is not default constructible (does not model forward iterator, JWV cannot be const-accessed)
    using Inner   = BasicVectorView<char, ViewProperties{.common = false}, EqComparableInputIter>;
    using V       = BasicVectorView<Inner, ViewProperties{.common = false}, forward_iterator>;
    using Pattern = BasicVectorView<char, ViewProperties{}, forward_iterator>;
    using JWV     = std::ranges::join_with_view<V, Pattern>;
    using Iter    = std::ranges::iterator_t<JWV>;

    Iter iter;
    assert(testEquality(iter, Iter{}, true));

    std::ranges::join_with_view<V, Pattern> jwv(V{Inner{'a', 'b'}, Inner{'c', 'd'}}, Pattern{',', ' '});
    Iter jwv_iter = jwv.begin();
    assert(testEquality(jwv_iter, iter, false));
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
