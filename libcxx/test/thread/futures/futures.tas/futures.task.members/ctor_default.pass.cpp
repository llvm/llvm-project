//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <future>

// class packaged_task<R(ArgTypes...)>

// packaged_task();

#include <future>
#include <cassert>

struct A {};

int main()
{
    std::packaged_task<A(int, char)> p;
    assert(!p.valid());
}
