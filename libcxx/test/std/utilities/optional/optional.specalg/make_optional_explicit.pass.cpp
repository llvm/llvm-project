//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// <optional>

// template <class T, class... Args>
//   constexpr optional<T> make_optional(Args&&... args);

// GCC crashes on this file, see https://gcc.gnu.org/bugzilla/show_bug.cgi?id=120577
// XFAIL: gcc-15

#include <cassert>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include "test_macros.h"

template <typename T>
constexpr bool test_ref() {
  T i{0};
  auto opt = std::make_optional<T&>(i);

#if TEST_STD_VER < 26
  assert((std::is_same_v<decltype(opt), std::optional<T>>));
#else
  assert((std::is_same_v<decltype(opt), std::optional<T&>>));
#endif

  assert(*opt == 0);

  return true;
}

int main(int, char**)
{
    {
        constexpr auto opt = std::make_optional<int>('a');
        static_assert(*opt == int('a'));
    }
    {
        std::string s = "123";
        auto opt = std::make_optional<std::string>(s);
        assert(*opt == "123");
    }
    {
        std::unique_ptr<int> s = std::make_unique<int>(3);
        auto opt = std::make_optional<std::unique_ptr<int>>(std::move(s));
        assert(**opt == 3);
        assert(s == nullptr);
    }
    {
        auto opt = std::make_optional<std::string>(4u, 'X');
        assert(*opt == "XXXX");
    }
    using namespace std::string_view_literals;

    static_assert(test_ref<int>());
    assert((test_ref<int>()));
    static_assert(test_ref<double>());
    assert((test_ref<double>()));

    return 0;
}
