//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

// ~flat_multimap();

#include <cassert>
#include <deque>
#include <flat_map>
#include <functional>
#include <type_traits>
#include <vector>

#include "test_macros.h"
#include "MoveOnly.h"
#include "test_allocator.h"

struct ThrowingDtorComp {
  constexpr bool operator()(const auto&, const auto&) const;
  constexpr ~ThrowingDtorComp() noexcept(false) {}
};

template <template <class...> class KeyContainer, template <class...> class ValueContainer>
constexpr void test() {
  {
    using C =
        std::flat_multimap<MoveOnly, MoveOnly, std::less<MoveOnly>, KeyContainer<MoveOnly>, ValueContainer<MoveOnly>>;
    static_assert(std::is_nothrow_destructible_v<C>);
  }
  {
    using V  = KeyContainer<MoveOnly, test_allocator<MoveOnly>>;
    using V2 = ValueContainer<MoveOnly, test_allocator<MoveOnly>>;
    using C  = std::flat_multimap<MoveOnly, MoveOnly, std::less<MoveOnly>, V, V2>;
    static_assert(std::is_nothrow_destructible_v<C>);
  }
  {
    using V  = KeyContainer<MoveOnly, test_allocator<MoveOnly>>;
    using V2 = ValueContainer<MoveOnly, test_allocator<MoveOnly>>;
    using C  = std::flat_multimap<MoveOnly, MoveOnly, std::greater<MoveOnly>, V, V2>;
    static_assert(std::is_nothrow_destructible_v<C>);
  }
#if defined(_LIBCPP_VERSION)
  {
    using C =
        std::flat_multimap<MoveOnly, MoveOnly, ThrowingDtorComp, KeyContainer<MoveOnly>, ValueContainer<MoveOnly>>;
    static_assert(!std::is_nothrow_destructible_v<C>);
  }
#endif // _LIBCPP_VERSION
}

void test() {
  test<std::vector, std::vector>();
  test<std::deque, std::deque>();
}
