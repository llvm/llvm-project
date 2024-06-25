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
  auto view = BasicTestView<SizedInputIterator>{SizedInputIterator(), SizedInputIterator()};
  auto sv = std::ranges::stride_view<BasicTestView<SizedInputIterator>>(view, 1);
  [[maybe_unused]] auto result = sv.begin().base();
  assert(result.move_counter==1);
  assert(result.copy_counter==0);
  return true;
}

int main(int, char**) {
  base_noexcept();
  static_assert(base_noexcept());
  base_const();
  static_assert(base_const());
  base_move();
  return 0;
}
