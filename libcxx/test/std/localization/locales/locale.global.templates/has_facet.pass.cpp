//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// template <class Facet> bool has_facet(const locale& loc) throw();

#include <locale>
#include <cassert>

struct my_facet
    : public std::locale::facet
{
    static std::locale::id id;
};

std::locale::id my_facet::id;

int main()
{
    std::locale loc;
    assert(std::has_facet<std::ctype<char> >(loc));
    assert(!std::has_facet<my_facet>(loc));
    std::locale loc2(loc, new my_facet);
    assert(std::has_facet<my_facet>(loc2));
}
