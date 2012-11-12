//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <vector>
// vector<bool>

// void resize(size_type sz);

#include <vector>
#include <cassert>

int main()
{
    {
        std::vector<bool> v(100);
        v.resize(50);
        assert(v.size() == 50);
        assert(v.capacity() >= 100);
        v.resize(200);
        assert(v.size() == 200);
        assert(v.capacity() >= 200);
    }
}
