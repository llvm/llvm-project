//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// XFAIL: availability-bad_optional_access-missing && !no-exceptions

// <optional>
//
// template <class T>
//   constexpr optional<decay_t<T>> make_optional(T&& v);

#include <optional>
#include <string>
#include <memory>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
        int arr[10];
        auto opt = std::make_optional(arr);
        ASSERT_SAME_TYPE(decltype(opt), std::optional<int*>);
        assert(*opt == arr);
    }
    {
        constexpr auto opt = std::make_optional(2);
        ASSERT_SAME_TYPE(decltype(opt), const std::optional<int>);
        static_assert(opt.value() == 2);
    }
    {
        auto opt = std::make_optional(2);
        ASSERT_SAME_TYPE(decltype(opt), std::optional<int>);
        assert(*opt == 2);
    }
    {
        const std::string s = "123";
        auto opt = std::make_optional(s);
        ASSERT_SAME_TYPE(decltype(opt), std::optional<std::string>);
        assert(*opt == "123");
    }
    {
        std::unique_ptr<int> s = std::make_unique<int>(3);
        auto opt = std::make_optional(std::move(s));
        ASSERT_SAME_TYPE(decltype(opt), std::optional<std::unique_ptr<int>>);
        assert(**opt == 3);
        assert(s == nullptr);
    }

  return 0;
}
