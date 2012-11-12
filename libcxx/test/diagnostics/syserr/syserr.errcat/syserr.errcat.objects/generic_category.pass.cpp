//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <system_error>

// class error_category

// const error_category& generic_category();

#include <system_error>
#include <cassert>
#include <string>

int main()
{
    const std::error_category& e_cat1 = std::generic_category();
    std::string m1 = e_cat1.name();
    assert(m1 == "generic");
}
