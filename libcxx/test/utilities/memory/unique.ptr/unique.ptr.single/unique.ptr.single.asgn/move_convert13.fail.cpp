//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <memory>

// unique_ptr

// Test unique_ptr converting move assignment

// Do not convert from an array unique_ptr

#include <memory>
#include <cassert>

struct A
{
};

struct Deleter
{
    void operator()(void*) {}
};

int main()
{
    std::unique_ptr<A[], Deleter> s;
    std::unique_ptr<A, Deleter> s2;
    s2 = std::move(s);
}
