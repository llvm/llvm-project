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

// test get

#include <memory>
#include <cassert>

int main()
{
    int* p = new int[3];
    std::unique_ptr<int[]> s(p);
    assert(s.get() == p);
}
