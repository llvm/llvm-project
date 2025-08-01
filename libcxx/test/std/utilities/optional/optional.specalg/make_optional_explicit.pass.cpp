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

#include <optional>
#include <string>
#include <memory>
#include <cassert>

#include "test_macros.h"

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

  return 0;
}
