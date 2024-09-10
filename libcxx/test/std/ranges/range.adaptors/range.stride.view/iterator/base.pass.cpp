//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr iterator_t<_Base> const& base() const& noexcept
// constexpr iterator_t<_Base> base() &&

#include <ranges>
#include <type_traits>

#include "../types.h"

constexpr bool base_noexcept() {
  {
    int arr[]                         = {1, 2, 3};
    auto stride                       = std::ranges::stride_view(arr, 1);
    [[maybe_unused]] auto stride_iter = stride.begin();

    // Check that calling base on an iterator where this is an lvalue reference
    // is noexcept.
    static_assert(noexcept(stride_iter.base()));
    // Calling base on an iterator where this is an rvalue reference may except.
    static_assert(!noexcept((std::move(stride_iter).base())));
  }

  return true;
}

constexpr bool base_const() {
  {
    int arr[]                         = {1, 2, 3};
    auto stride                       = std::ranges::stride_view(arr, 1);
    [[maybe_unused]] auto stride_iter = stride.begin();

    // Calling base on an iterator where this is lvalue returns a const ref to an iterator.
    static_assert(std::is_const_v<std::remove_reference_t<decltype(stride_iter.base())>>);
    // Calling base on an iterator where this is an rvalue reference returns a non-const iterator.
    static_assert(!std::is_const_v<decltype(std::move(stride_iter).base())>);
  }

  return true;
}

bool base_move() {
  // Keep track of how many times the original iterator is moved
  // and/or copied during the test.
  int move_counter = 0;
  int copy_counter = 0;

  auto start         = SizedInputIter();
  start.move_counter = &move_counter;
  start.copy_counter = &copy_counter;
  auto stop          = SizedInputIter();

  auto view = BasicTestView<SizedInputIter>{start, stop};
  auto sv   = std::ranges::stride_view<BasicTestView<SizedInputIter>>(view, 1);
  auto svi  = sv.begin();

  // Reset the move/copy counters so that they reflect *only* whether the
  // base() member function moved or copied the iterator.
  move_counter                 = 0;
  copy_counter                 = 0;
  [[maybe_unused]] auto result = std::move(svi).base();

  // Ensure that base std::move'd the iterator.
  assert(*result.move_counter == 1);
  assert(*result.copy_counter == 0);
  return true;
}

bool base_copy() {
  // See above.
  int move_counter = 0;
  int copy_counter = 0;
  auto start       = SizedInputIter();

  start.move_counter = &move_counter;
  start.copy_counter = &copy_counter;
  auto stop          = SizedInputIter();

  auto view                 = BasicTestView<SizedInputIter>{start, stop};
  auto sv                   = std::ranges::stride_view<BasicTestView<SizedInputIter>>(view, 1);
  [[maybe_unused]] auto svi = sv.begin();

  // See above.
  move_counter                                      = 0;
  copy_counter                                      = 0;
  [[maybe_unused]] const SizedInputIter& result = svi.base();

  // Ensure that base did _not_ std::move'd the iterator.
  assert(*result.move_counter == 0);
  assert(*result.copy_counter == 0);
  return true;
}

int main(int, char**) {
  base_noexcept();
  static_assert(base_noexcept());
  base_const();
  static_assert(base_const());
  base_move();
  base_copy();
  return 0;
}
