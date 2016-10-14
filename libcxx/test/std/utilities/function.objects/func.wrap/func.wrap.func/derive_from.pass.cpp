//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03
// REQUIRES-ANY: c++11, c++14

// <functional>

// See https://llvm.org/bugs/show_bug.cgi?id=20002

#include <functional>
#include <type_traits>

#include "test_macros.h"

using Fn = std::function<void()>;
struct S : public std::function<void()> { using function::function; };

int main() {
    S s( [](){} );
    S f1( s );
    S f2(std::allocator_arg, std::allocator<int>{}, s);
}
