//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14
// <optional>

// template <class T, class... Args>
//   constexpr optional<T> make_optional(Args&&... args);

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
        constexpr auto opt = make_optional<int>('a');
        static_assert(*opt == int('a'), "");
    }
    {
        std::string s("123");
        auto opt = make_optional<std::string>(s);
        assert(*opt == s);
    }
    {
        std::unique_ptr<int> s(new int(3));
        auto opt = make_optional<std::unique_ptr<int>>(std::move(s));
        assert(**opt == 3);
        assert(s == nullptr);
    }
    {
        auto opt = make_optional<std::string>(4, 'X');
        assert(*opt == "XXXX");
    }

  return 0;
}
