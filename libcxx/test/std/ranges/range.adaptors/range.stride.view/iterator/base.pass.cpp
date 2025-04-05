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

constexpr bool base_move() {
  // Keep track of how many times the original iterator is moved
  // and/or copied during the test.
  int move_counter = 0;
  int copy_counter = 0;

  auto start         = SizedInputIter();
  start.move_counter = &move_counter;
  start.copy_counter = &copy_counter;
  auto stop          = SizedInputIter();

  auto view = BasicTestView<SizedInputIter>{start, stop};
  assert(move_counter == 0);
  // One copies of _start_ occurs when it is copied to the basic test view's member variable.
  assert(copy_counter == 1);

  auto sv = std::ranges::stride_view<BasicTestView<SizedInputIter>>(view, 1);
  // There is a copy of _view_ made when it is passed by value.
  // There is a move done of _view_ when it is used as the initial value of __base.
  assert(move_counter == 1);
  assert(copy_counter == 2);

  auto svi = sv.begin();
  // Another copy of _start_ when begin uses the iterator to the first element
  // of the view underlying sv as the by-value parameter to the stride view iterator's
  // constructor.
  assert(copy_counter == 3);
  // Another move of __start_ happens right after that when it is std::move'd to
  // become the first __current of the iterator.
  assert(move_counter == 2);

  [[maybe_unused]] auto result = std::move(svi).base();
  // Ensure that base std::move'd the iterator and did not copy it.
  assert(move_counter == 3);
  assert(copy_counter == 3);
  return true;
}

constexpr bool base_copy() {
  // See base_move() for complete description of when/why
  // moves/copies take place..
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

  [[maybe_unused]] const SizedInputIter result = svi.base();
  // Ensure that base did _not_ std::move'd the iterator.
  assert(move_counter == 2);
  assert(copy_counter == 4);

  return true;
}

int main(int, char**) {
  base_noexcept();
  static_assert(base_noexcept());

  base_const();
  static_assert(base_const());

  base_move();
  static_assert(base_move());

  base_copy();
  static_assert(base_copy());
  return 0;
}
