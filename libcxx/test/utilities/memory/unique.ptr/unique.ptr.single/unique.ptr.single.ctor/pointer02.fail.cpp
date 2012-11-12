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

// Test unique_ptr(pointer) ctor

#include <memory>

// unique_ptr(pointer) ctor should require non-reference Deleter ctor
class Deleter
{
public:

    void operator()(void*) {}
};

int main()
{
    std::unique_ptr<int, Deleter&> p(new int);
}
