//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14
// <optional>

// constexpr optional(const optional<T>&& rhs);
//   If is_trivially_move_constructible_v<T> is true,
//    this constructor shall be a constexpr constructor.

#include <optional>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

struct S {
    constexpr S()   : v_(0) {}
    S(int v)        : v_(v) {}
    constexpr S(const S  &rhs) : v_(rhs.v_) {} // not trivially moveable
    constexpr S(const S &&rhs) : v_(rhs.v_) {} // not trivially moveable
    int v_;
    };


int main(int, char**)
{
    static_assert (!std::is_trivially_move_constructible_v<S>, "" );
    constexpr std::optional<S> o1;
    constexpr std::optional<S> o2 = std::move(o1);  // not constexpr

  return 0;
}
