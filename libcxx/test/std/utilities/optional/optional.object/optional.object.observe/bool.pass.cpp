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
// constexpr explicit optional<T&>::operator bool() const noexcept;

#include <cassert>
#include <optional>
#include <type_traits>

#include "test_macros.h"

#if TEST_STD_VER >= 26

constexpr bool test_ref() {
  {
    std::optional<int&> opt;
    ASSERT_NOEXCEPT(bool(opt));
    assert(!opt);
    static_assert(!std::is_convertible<std::optional<int&>, bool>::value, "");
  }
  {
    int i = 1;
    std::optional<int&> opt(i);
    ASSERT_NOEXCEPT(bool(opt));
    assert(opt);
    static_assert(!std::is_convertible<std::optional<int&>, bool>::value, "");
  }

  return true;
}

#endif

int main(int, char**)
{
    using std::optional;
    {
        const optional<int> opt; ((void)opt);
        ASSERT_NOEXCEPT(bool(opt));
        static_assert(!std::is_convertible<optional<int>, bool>::value, "");
    }
    {
        constexpr optional<int> opt;
        static_assert(!opt, "");
    }
    {
        constexpr optional<int> opt(0);
        static_assert(opt, "");
    }

#if TEST_STD_VER >= 26
    {
      assert(test_ref());
      static_assert(test_ref());
    }
#endif

    return 0;
}
