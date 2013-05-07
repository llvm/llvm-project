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

// test release

#include <memory>
#include <cassert>

int main()
{
    std::unique_ptr<int[]> p(new int[3]);
    int* i = p.get();
    int* j = p.release();
    assert(p.get() == 0);
    assert(i == j);
    delete [] j;
}
