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

// A program that necessitates the instantiation of template optional for a
// reference type is ill-formed.

#include <experimental/optional>

int main()
{
    using std::experimental::optional;

    optional<int&> opt;
}
