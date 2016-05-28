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

// T shall be an object type and shall satisfy the requirements of Destructible

#include <experimental/optional>

int main()
{
    using std::experimental::optional;

    optional<void> opt;
}
