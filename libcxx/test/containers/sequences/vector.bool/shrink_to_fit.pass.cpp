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

// void shrink_to_fit();

#include <vector>
#include <cassert>

int main()
{
    {
        std::vector<bool> v(100);
        v.push_back(1);
        v.shrink_to_fit();
        assert(v.capacity() >= 101);
        assert(v.size() >= 101);
    }
}
