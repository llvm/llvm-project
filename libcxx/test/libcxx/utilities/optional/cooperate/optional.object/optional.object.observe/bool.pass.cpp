//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// <optional>

// constexpr explicit optional<T>::operator bool() const noexcept;

#include <optional>
#include <functional>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

using std::optional;

template <typename T, bool ConstexprInit = true, typename F>
constexpr bool test(F init) {
  {
    const optional<T> opt;
    (void)opt;
    ASSERT_NOEXCEPT(bool(opt));
    static_assert(!std::is_convertible_v<optional<T>, bool>);
  }
  {
    constexpr optional<T> opt;
    static_assert(!opt);
  }
  if constexpr (ConstexprInit) {
    constexpr optional<T> opt(init());
    static_assert(opt);
  }
  return true;
}

int f() { return 0; }

int main(int, char**) {
  {
    static int i;
    constexpr bool constructor_is_constexpr = TEST_STD_VER >= 20;
    test<std::reference_wrapper<int>, constructor_is_constexpr>([]() -> auto& { return i; });
    test<std::reference_wrapper<int()>, constructor_is_constexpr>([]() -> auto& { return f; });
  }
}
