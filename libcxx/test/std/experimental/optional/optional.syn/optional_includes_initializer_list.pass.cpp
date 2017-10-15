//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11
// <optional>

// #include <initializer_list>

#include <experimental/optional>

int main()
{
    using std::experimental::optional;

    std::initializer_list<int> list;
    (void)list;
}
