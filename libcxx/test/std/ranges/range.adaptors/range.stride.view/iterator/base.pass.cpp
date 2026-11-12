//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr const iterator_t<Base>& base() const& noexcept
// constexpr iterator_t<Base> base() &&

#include <cassert>
#include <ranges>
#include <type_traits>
#include <utility>

#include "../types.h"

constexpr bool test() {
  {
    // base() const& is noexcept; base() && is not.
    int arr[]                         = {1, 2, 3};
    auto stride                       = std::ranges::stride_view(arr, 1);
    [[maybe_unused]] auto stride_iter = stride.begin();

    static_assert(noexcept(stride_iter.base()));
    static_assert(!noexcept((std::move(stride_iter).base())));
  }
  {
    // base() const& returns a const ref; base() && returns a non-const value.
    int arr[]                         = {1, 2, 3};
    auto stride                       = std::ranges::stride_view(arr, 1);
    [[maybe_unused]] auto stride_iter = stride.begin();

    static_assert(std::is_const_v<std::remove_reference_t<decltype(stride_iter.base())>>);
    static_assert(!std::is_const_v<decltype(std::move(stride_iter).base())>);
  }
  {
    // base() && moves the underlying iterator.
    int move_counter = 0;
    int copy_counter = 0;

    auto start         = SizedInputIter();
    start.move_counter = &move_counter;
    start.copy_counter = &copy_counter;
    auto stop          = SizedInputIter();

    auto view = BasicTestView<SizedInputIter>{start, stop};
    assert(move_counter == 0);
    assert(copy_counter == 1);

    auto sv = std::ranges::stride_view<BasicTestView<SizedInputIter>>(view, 1);
    assert(move_counter == 1);
    assert(copy_counter == 2);

    auto svi = sv.begin();
    assert(copy_counter == 3);
    assert(move_counter == 2);

    [[maybe_unused]] auto result = std::move(svi).base();
    assert(move_counter == 3);
    assert(copy_counter == 3);
  }
  {
    // base() const& copies the underlying iterator.
    int move_counter = 0;
    int copy_counter = 0;
    auto start       = SizedInputIter();

    start.move_counter = &move_counter;
    start.copy_counter = &copy_counter;
    auto stop          = SizedInputIter();

    auto view = BasicTestView<SizedInputIter>{start, stop};
    assert(move_counter == 0);
    assert(copy_counter == 1);

    auto sv = std::ranges::stride_view<BasicTestView<SizedInputIter>>(view, 1);
    assert(move_counter == 1);
    assert(copy_counter == 2);

    [[maybe_unused]] auto svi = sv.begin();
    assert(copy_counter == 3);
    assert(move_counter == 2);

    [[maybe_unused]] const SizedInputIter base_result = svi.base();
    assert(move_counter == 2);
    assert(copy_counter == 4);
  }
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
