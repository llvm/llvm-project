//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <chrono>
// class weekday;

//  constexpr bool ok() const noexcept;
//  Returns: wd_ <= 6

#include <chrono>
#include <cassert>
#include <type_traits>
#include <utility>

#include "test_macros.h"

int main(int, char**)
{
    using weekday = std::chrono::weekday;

    ASSERT_NOEXCEPT(                std::declval<const weekday>().ok());
    ASSERT_SAME_TYPE(bool, decltype(std::declval<const weekday>().ok()));

    static_assert( weekday{0}.ok(), "");
    static_assert( weekday{1}.ok(), "");
    static_assert( weekday{7}.ok(), "");  // 7 is transmogrified into 0 in the ctor
    static_assert(!weekday{8}.ok(), "");

    for (unsigned i = 0; i <= 7; ++i)
        assert(weekday{i}.ok());
    for (unsigned i = 8; i <= 255; ++i)
        assert(!weekday{i}.ok());

  return 0;
}
