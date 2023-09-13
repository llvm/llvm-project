//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <ios>

// const error_category& iostream_category();

#include <ios>
#include <cassert>
#include <string>

#include "test_macros.h"

// See https://llvm.org/D65667
struct StaticInit {
    const std::error_category* ec;
    ~StaticInit() {
        std::string str = ec->name();
        assert(str == "iostream") ;
    }
};
static StaticInit foo;

int main(int, char**)
{
    {
        const std::error_category& e_cat1 = std::iostream_category();
        std::string m1 = e_cat1.name();
        assert(m1 == "iostream");
    }

    {
        foo.ec = &std::iostream_category();
        std::string m2 = foo.ec->name();
        assert(m2 == "iostream");
    }

    return 0;
}
