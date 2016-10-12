//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14
// <optional>

// struct nullopt_t{see below};

#include <optional>

using std::optional;
using std::nullopt_t;

int main()
{
    // I roughly interpret LWG2736 as "it shall not be possible to copy-list-initialize nullopt_t with an
    // empty braced-init-list."
    nullopt_t foo = {};
}
