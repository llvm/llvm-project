//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// class money_base
// {
// public:
//     enum part {none, space, symbol, sign, value};
//     struct pattern {char field[4];};
// };

#include <locale>
#include <cassert>

int main()
{
    std::money_base mb; ((void)mb);
    static_assert(std::money_base::none == 0, "");
    static_assert(std::money_base::space == 1, "");
    static_assert(std::money_base::symbol == 2, "");
    static_assert(std::money_base::sign == 3, "");
    static_assert(std::money_base::value == 4, "");
    static_assert(sizeof(std::money_base::pattern) == 4, "");
    std::money_base::pattern p;
    p.field[0] = std::money_base::none;
}
