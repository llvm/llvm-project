//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <vector>

// const_pointer data() const;

#include <vector>
#include <cassert>

int main()
{
    {
        const std::vector<int> v;
        assert(v.data() == 0);
    }
    {
        const std::vector<int> v(100);
        assert(v.data() == &v.front());
    }
}
