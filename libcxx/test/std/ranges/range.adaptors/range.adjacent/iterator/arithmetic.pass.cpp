//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr iterator& operator+=(difference_type x)
//   requires random_access_range<Base>;
// constexpr iterator& operator-=(difference_type x)
//   requires random_access_range<Base>;
// friend constexpr iterator operator+(const iterator& i, difference_type n)
//   requires random_access_range<Base>;
// friend constexpr iterator operator+(difference_type n, const iterator& i)
//   requires random_access_range<Base>;
// friend constexpr iterator operator-(const iterator& i, difference_type n)
//   requires random_access_range<Base>;
// friend constexpr difference_type operator-(const iterator& x, const iterator& y)
//   requires sized_sentinel_for<iterator_t<Base>, iterator_t<Base>>;

#include <cassert>
#include <iterator>
#include <ranges>

#include <array>
#include <concepts>
#include <functional>

#include "../../range_adaptor_types.h"

template <class T, class U>
concept canPlusEqual = requires(T& t, U& u) { t += u; };

template <class T, class U>
concept canMinusEqual = requires(T& t, U& u) { t -= u; };

template <class R, std::size_t N>
constexpr void test() {
  int buffer[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};

  const auto validateRefFromIndex = [&](auto&& tuple, std::size_t idx) {
    assert(&std::get<0>(tuple) == &buffer[idx]);
    if constexpr (N >= 2)
      assert(&std::get<1>(tuple) == &buffer[idx + 1]);
    if constexpr (N >= 3)
      assert(&std::get<2>(tuple) == &buffer[idx + 2]);
    if constexpr (N >= 4)
      assert(&std::get<3>(tuple) == &buffer[idx + 3]);
    if constexpr (N >= 5)
      assert(&std::get<4>(tuple) == &buffer[idx + 4]);
  };

  R rng{buffer};
  {
    // operator+(x, n) and operator+=
    std::ranges::adjacent_view<R, N> v(rng);
    auto it1 = v.begin();

    validateRefFromIndex(*it1, 0);

    auto it2 = it1 + 3;
    validateRefFromIndex(*it2, 3);

    auto it3 = 3 + it1;
    validateRefFromIndex(*it3, 3);

    it1 += 3;
    assert(it1 == it2);
    validateRefFromIndex(*it1, 3);
  }

  {
    // operator-(x, n) and operator-=
    std::ranges::adjacent_view<R, N> v(rng);
    auto it1 = v.end();

    auto it2 = it1 - 3;
    validateRefFromIndex(*it2, 7 - N);

    it1 -= 3;
    assert(it1 == it2);
    validateRefFromIndex(*it1, 7 - N);
  }

  {
    // operator-(x, y)
    std::ranges::adjacent_view<R, N> v(rng);
    assert((v.end() - v.begin()) == (10 - N));

    auto it1 = v.begin() + 2;
    auto it2 = v.end() - 1;
    assert((it1 - it2) == static_cast<long>(N) - 7);
  }

  {
    // empty range
    std::ranges::adjacent_view<R, N> v(R{buffer, 0});
    assert((v.end() - v.begin()) == 0);
  }

  {
    // N > size of range
    std::ranges::adjacent_view<R, 3> v(R{buffer, 2});
    assert((v.end() - v.begin()) == 0);
  }
}

template <std::size_t N>
constexpr void test() {
  test<ContiguousCommonView, N>();
  test<SimpleCommonRandomAccessSized, N>();

  {
    // Non random access but sized sentinel
    int buffer[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    using View   = std::ranges::adjacent_view<ForwardSizedView, N>;
    using Iter   = std::ranges::iterator_t<View>;
    using Diff   = std::iter_difference_t<Iter>;

    static_assert(!std::invocable<std::plus<>, Iter, Diff>);
    static_assert(!std::invocable<std::plus<>, Diff, Iter>);
    static_assert(!canPlusEqual<Iter, Diff>);
    static_assert(!std::invocable<std::minus<>, Iter, Diff>);
    static_assert(!canMinusEqual<Iter, Diff>);
    static_assert(std::invocable<std::minus<>, Iter, Iter>);

    View v(ForwardSizedView{buffer});
    auto it1 = v.begin();
    auto it2 = v.end();
    assert((it2 - it1) == (10 - N));
  }

  {
    // Non random access and non-sized sentinel
    using View = std::ranges::adjacent_view<SimpleNonCommonNonRandom, N>;
    using Iter = std::ranges::iterator_t<View>;
    using Diff = std::iter_difference_t<Iter>;

    static_assert(!std::invocable<std::plus<>, Iter, Diff>);
    static_assert(!std::invocable<std::plus<>, Diff, Iter>);
    static_assert(!canPlusEqual<Iter, Diff>);
    static_assert(!std::invocable<std::minus<>, Iter, Diff>);
    static_assert(!canMinusEqual<Iter, Diff>);
    static_assert(!std::invocable<std::minus<>, Iter, Iter>);
  }
}

constexpr bool test() {
  test<1>();
  test<2>();
  test<3>();
  test<4>();
  test<5>();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
