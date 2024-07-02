//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <ranges>

// friend constexpr void iter_swap(const iterator& x, const iterator& y)
//   requires indirectly_swappable<InnerIter, PatternIter> {
//   visit(ranges::iter_swap, x.inner_it_, y.inner_it_);
// }

#include <ranges>

#include <algorithm>
#include <cassert>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

template <class I>
concept CanIterSwap = requires(I i) { iter_swap(i); };

constexpr bool test() {
  { // Test common usage
    using V       = std::vector<std::string>;
    using Pattern = std::string;
    using JWV     = std::ranges::join_with_view<std::ranges::owning_view<V>, std::ranges::owning_view<Pattern>>;
    using namespace std::string_view_literals;

    JWV jwv(V{"std", "ranges", "views", "join_with_view"}, Pattern{":: "});
    assert(std::ranges::equal(jwv, "std:: ranges:: views:: join_with_view"sv));

    auto it = jwv.begin();
    iter_swap(it, std::ranges::next(it, 2)); // Swap elements of the same inner range.
    assert(std::ranges::equal(jwv, "dts:: ranges:: views:: join_with_view"sv));

    std::ranges::advance(it, 3);
    iter_swap(std::as_const(it), std::ranges::next(it, 2)); // Swap elements of the pattern.
    assert(std::ranges::equal(jwv, "dts ::ranges ::views ::join_with_view"sv));

    std::ranges::advance(it, 3);
    const auto it2 = jwv.begin();
    iter_swap(std::as_const(it), it2); // Swap elements of different inner ranges.
    assert(std::ranges::equal(jwv, "rts ::danges ::views ::join_with_view"sv));

    std::ranges::advance(it, 6);
    iter_swap(std::as_const(it), it2); // Swap element from inner range with element from the pattern.
    assert(std::ranges::equal(jwv, " tsr::dangesr::viewsr::join_with_view"sv));

    static_assert(std::is_void_v<decltype(iter_swap(it, it))>);
    static_assert(std::is_void_v<decltype(iter_swap(it2, it2))>);
    static_assert(!CanIterSwap<std::ranges::iterator_t<const JWV>>);
    static_assert(!CanIterSwap<const std::ranges::iterator_t<const JWV>>);
  }

  { // InnerIter and PatternIter don't model indirectly swappable
    using JWV = std::ranges::join_with_view<std::span<std::string>, std::string_view>;
    static_assert(!CanIterSwap<std::ranges::iterator_t<JWV>>);
    static_assert(!CanIterSwap<const std::ranges::iterator_t<JWV>>);
    static_assert(!CanIterSwap<std::ranges::iterator_t<const JWV>>);
    static_assert(!CanIterSwap<const std::ranges::iterator_t<const JWV>>);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
