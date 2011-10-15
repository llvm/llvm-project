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

#include <memory>

// default unique_ptr ctor should require non-pointer Deleter

int main()
{
    std::unique_ptr<int, void (*)(void*)> p;
}
