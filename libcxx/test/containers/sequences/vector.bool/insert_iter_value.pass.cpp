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

// iterator insert(const_iterator position, const value_type& x);

#include <vector>
#include <cassert>

int main()
{
    {
        std::vector<bool> v(100);
        std::vector<bool>::iterator i = v.insert(v.cbegin() + 10, 1);
        assert(v.size() == 101);
        assert(i == v.begin() + 10);
        int j;
        for (j = 0; j < 10; ++j)
            assert(v[j] == 0);
        assert(v[j] == 1);
        for (++j; j < 101; ++j)
            assert(v[j] == 0);
    }
}
