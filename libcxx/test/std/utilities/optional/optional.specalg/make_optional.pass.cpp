//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// Throwing bad_optional_access is supported starting in macosx10.13
// XFAIL: use_system_cxx_lib && x86_64-apple-macosx10.12 && !no-exceptions
// XFAIL: use_system_cxx_lib && x86_64-apple-macosx10.11 && !no-exceptions
// XFAIL: use_system_cxx_lib && x86_64-apple-macosx10.10 && !no-exceptions
// XFAIL: use_system_cxx_lib && x86_64-apple-macosx10.9 && !no-exceptions

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
    using std::optional;
    using std::make_optional;
    {
        int arr[10]; ((void)arr);
        ASSERT_SAME_TYPE(decltype(make_optional(arr)), optional<int*>);
    }
    {
        constexpr auto opt = make_optional(2);
        ASSERT_SAME_TYPE(decltype(opt), const optional<int>);
        static_assert(opt.value() == 2);
    }
    {
        optional<int> opt = make_optional(2);
        assert(*opt == 2);
    }
    {
        std::string s("123");
        optional<std::string> opt = make_optional(s);
        assert(*opt == s);
    }
    {
        std::unique_ptr<int> s(new int(3));
        optional<std::unique_ptr<int>> opt = make_optional(std::move(s));
        assert(**opt == 3);
        assert(s == nullptr);
    }

  return 0;
}
