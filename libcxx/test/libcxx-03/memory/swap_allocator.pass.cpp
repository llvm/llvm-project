//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//

// <memory>

// template <typename _Alloc>
// void __swap_allocator(_Alloc& __a1, _Alloc& __a2);

#include <__cxx03/__memory/swap_allocator.h>
#include <cassert>
#include <memory>
#include <utility>

#include "test_macros.h"

template <bool Propagate, bool Noexcept>
struct Alloc {
  int i = 0;
  Alloc() = default;
  Alloc(int set_i) : i(set_i) {}

  using value_type = int;
  using propagate_on_container_swap = std::integral_constant<bool, Propagate>;

  friend void swap(Alloc& a1, Alloc& a2) TEST_NOEXCEPT_COND(Noexcept) {
    std::swap(a1.i, a2.i);
  }

};

using PropagatingAlloc = Alloc</*Propagate=*/true, /*Noexcept=*/true>;
static_assert(std::allocator_traits<PropagatingAlloc>::propagate_on_container_swap::value, "");

using NonPropagatingAlloc = Alloc</*Propagate=*/false, /*Noexcept=*/true>;
static_assert(!std::allocator_traits<NonPropagatingAlloc>::propagate_on_container_swap::value, "");

using NoexceptSwapAlloc = Alloc</*Propagate=*/true, /*Noexcept=*/true>;
using ThrowingSwapAlloc = Alloc</*Propagate=*/true, /*Noexcept=*/false>;

int main(int, char**) {
  {
    PropagatingAlloc a1(1), a2(42);
    std::__swap_allocator(a1, a2);
    assert(a1.i == 42);
    assert(a2.i == 1);
  }

  {
    NonPropagatingAlloc a1(1), a2(42);
    std::__swap_allocator(a1, a2);
    assert(a1.i == 1);
    assert(a2.i == 42);
  }

  return 0;
}
