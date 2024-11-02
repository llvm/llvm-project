//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads

// <future>

// enum class launch
// {
//     async = 1,
//     deferred = 2,
//     any = async | deferred /* EXTENSION */
// };

#include <future>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
#if TEST_STD_VER < 11
   LIBCPP_STATIC_ASSERT(static_cast<int>(std::launch::any) ==
                 (static_cast<int>(std::launch::async) | static_cast<int>(std::launch::deferred)), "");
#else
    LIBCPP_STATIC_ASSERT(std::launch::any == (std::launch::async | std::launch::deferred), "");
    static_assert(std::launch(0) == (std::launch::async & std::launch::deferred), "");
    LIBCPP_STATIC_ASSERT(std::launch::any == (std::launch::async ^ std::launch::deferred), "");
    LIBCPP_STATIC_ASSERT(std::launch::deferred == ~std::launch::async, "");
    std::launch x = std::launch::async;
    x &= std::launch::deferred;
    assert(x == std::launch(0));
    x = std::launch::async;
    x |= std::launch::deferred;
    LIBCPP_ASSERT(x == std::launch::any);
    x ^= std::launch::deferred;
    assert(x == std::launch::async);
#endif
    static_assert(static_cast<int>(std::launch::async) == 1, "");
    static_assert(static_cast<int>(std::launch::deferred) == 2, "");

  return 0;
}
