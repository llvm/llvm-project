//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
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
    std::money_base mb;
    assert(mb.none == 0);
    assert(mb.space == 1);
    assert(mb.symbol == 2);
    assert(mb.sign == 3);
    assert(mb.value == 4);
    assert(sizeof(std::money_base::pattern) == 4);
    std::money_base::pattern p;
    p.field[0] = std::money_base::none;
}
