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

// void push_back(const value_type& x);

#include <vector>
#include <cassert>

int main()
{
    {
        bool a[] = {0, 1, 1, 0, 1, 0, 0};
        const unsigned N = sizeof(a)/sizeof(a[0]);
        std::vector<int> c;
        for (unsigned i = 0; i < N; ++i)
        {
            c.push_back(a[i]);
            assert(c.size() == i+1);
            for (int j = 0; j < c.size(); ++j)
                assert(c[j] == a[j]);
        }
    }
}
