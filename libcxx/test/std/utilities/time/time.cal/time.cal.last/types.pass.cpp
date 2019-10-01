//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++98, c++03, c++11, c++14, c++17

// <chrono>

// struct last_spec {
//   explicit last_spec() = default;
// };
//
// inline constexpr last_spec last{};

#include <chrono>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    using last_spec = std::chrono::last_spec;

    ASSERT_SAME_TYPE(const last_spec, decltype(std::chrono::last));

    static_assert(std::is_trivially_copyable_v<last_spec>, "");
    static_assert(std::is_standard_layout_v<last_spec>, "");

  return 0;
}
