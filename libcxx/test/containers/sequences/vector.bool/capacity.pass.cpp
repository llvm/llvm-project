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

// size_type capacity() const;

#include <vector>
#include <cassert>

int main()
{
    {
        std::vector<bool> v;
        assert(v.capacity() == 0);
    }
    {
        std::vector<bool> v(100);
        assert(v.capacity() >= 100);
        v.push_back(0);
        assert(v.capacity() >= 101);
    }
}
