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

// Test unique_ptr default ctor

// default unique_ptr ctor should require non-reference Deleter ctor

#include <memory>

class Deleter
{
public:

    void operator()(void*) {}
};

int main()
{
    std::unique_ptr<int[], Deleter&> p;
}
