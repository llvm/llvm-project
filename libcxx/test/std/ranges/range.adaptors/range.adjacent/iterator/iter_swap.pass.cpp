//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// friend constexpr void iter_swap(const iterator& l, const iterator& r) noexcept(see below)
//   requires indirectly_swappable<iterator_t<Base>>;

#include <array>
#include <cassert>
#include <cstddef>
#include <ranges>

#include "../helpers.h"
#include "../../range_adaptor_types.h"

struct ThrowingMove {
  ThrowingMove() = default;
  ThrowingMove(ThrowingMove&&) {}
  ThrowingMove& operator=(ThrowingMove&&) { return *this; }
};

template <std::size_t N>
constexpr void test() {
  {
    // underlying iter_swap noexcept
    int buffer[] = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    };

    auto v = buffer | std::views::adjacent<N>;

    auto iter1 = v.begin();
    auto iter2 = v.begin() + 10;
    static_assert(noexcept(std::ranges::iter_swap(iter1, iter2)));

    std::ranges::iter_swap(iter1, iter2);

    assert(buffer[0] == 11);
    assert(buffer[10] == 1);
    if constexpr (N >= 2) {
      assert(buffer[1] == 12);
      assert(buffer[11] == 2);
    }
    if constexpr (N >= 3) {
      assert(buffer[2] == 13);
      assert(buffer[12] == 3);
    }
    if constexpr (N >= 4) {
      assert(buffer[3] == 14);
      assert(buffer[13] == 4);
    }
    if constexpr (N >= 5) {
      assert(buffer[4] == 15);
      assert(buffer[14] == 5);
    }

    auto tuple1 = *iter1;
    auto tuple2 = *iter2;
    assert(&std::get<0>(tuple1) == &buffer[0]);
    assert(&std::get<0>(tuple2) == &buffer[10]);

    if constexpr (N >= 2) {
      assert(&std::get<1>(tuple1) == &buffer[1]);
      assert(&std::get<1>(tuple2) == &buffer[11]);
    }
    if constexpr (N >= 3) {
      assert(&std::get<2>(tuple1) == &buffer[2]);
      assert(&std::get<2>(tuple2) == &buffer[12]);
    }
    if constexpr (N >= 4) {
      assert(&std::get<3>(tuple1) == &buffer[3]);
      assert(&std::get<3>(tuple2) == &buffer[13]);
    }
    if constexpr (N >= 5) {
      assert(&std::get<4>(tuple1) == &buffer[4]);
      assert(&std::get<4>(tuple2) == &buffer[14]);
    }
  }

  {
    // underlying iter_swap may throw
    std::array<ThrowingMove, 10> iterSwapMayThrow{};
    auto v     = iterSwapMayThrow | std::views::adjacent<N>;
    auto iter1 = v.begin();
    auto iter2 = ++v.begin();
    static_assert(!noexcept(std::ranges::iter_swap(iter1, iter2)));
  }

  {
    // underlying iterators' iter_swap are called through ranges::iter_swap
    auto rng = adltest::IterMoveSwapRange{};
    auto v   = rng | std::views::adjacent<N>;
    assert(rng.iter_move_called_times == 0);
    auto it1 = v.begin();
    auto it2 = std::ranges::next(it1, 3);

    std::ranges::iter_swap(it1, it2);
    assert(rng.iter_swap_called_times == 2 * N);

    std::ranges::iter_swap(it1, it2);
    assert(rng.iter_swap_called_times == 4 * N);
  }

  {
    // !indirectly_swappable<iterator_t<Base>>;

    const int buffer[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto v             = buffer | std::views::adjacent<N>;
    auto it1           = v.begin();
    auto it2           = v.begin() + 1;
    static_assert(!std::invocable<decltype(std::ranges::iter_swap), decltype(it1), decltype(it2)>);
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
