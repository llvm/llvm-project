//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <variant>

// template <class T, class... Types>
// constexpr bool holds_alternative(const variant<Types...>& v) noexcept;

#include "test_macros.h"
#include <variant>

int main(int, char**) {
  {
    using V = std::variant<int>;
    constexpr V v;
    static_assert(std::holds_alternative<int>(v), "");
  }
  {
    using V = std::variant<int, long>;
    constexpr V v;
    static_assert(std::holds_alternative<int>(v), "");
    static_assert(!std::holds_alternative<long>(v), "");
  }
  { // noexcept test
    using V = std::variant<int>;
    const V v;
    ASSERT_NOEXCEPT(std::holds_alternative<int>(v));
  }

  return 0;
}
