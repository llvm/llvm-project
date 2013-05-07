//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// test forward

#include <utility>

struct A
{
};

A source() {return A();}
const A csource() {return A();}

int main()
{
    std::forward<A&>(source());  // error
}
