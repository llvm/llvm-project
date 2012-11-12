//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <set>

// class multiset

//       iterator begin();
// const_iterator begin() const;
//       iterator end();
// const_iterator end()   const;
//
//       reverse_iterator rbegin();
// const_reverse_iterator rbegin() const;
//       reverse_iterator rend();
// const_reverse_iterator rend()   const;
//
// const_iterator         cbegin()  const;
// const_iterator         cend()    const;
// const_reverse_iterator crbegin() const;
// const_reverse_iterator crend()   const;

#include <set>
#include <cassert>

int main()
{
    {
        typedef int V;
        V ar[] =
        {
            1,
            1,
            1,
            2,
            2,
            2,
            3,
            3,
            3,
            4,
            4,
            4,
            5,
            5,
            5,
            6,
            6,
            6,
            7,
            7,
            7,
            8,
            8,
            8
        };
        std::multiset<int> m(ar, ar+sizeof(ar)/sizeof(ar[0]));
        assert(std::distance(m.begin(), m.end()) == m.size());
        assert(std::distance(m.rbegin(), m.rend()) == m.size());
        std::multiset<int>::iterator i;
        i = m.begin();
        std::multiset<int>::const_iterator k = i;
        assert(i == k);
        for (int j = 1; j <= 8; ++j)
            for (int k = 0; k < 3; ++k, ++i)
                assert(*i == j);
    }
    {
        typedef int V;
        V ar[] =
        {
            1,
            1,
            1,
            2,
            2,
            2,
            3,
            3,
            3,
            4,
            4,
            4,
            5,
            5,
            5,
            6,
            6,
            6,
            7,
            7,
            7,
            8,
            8,
            8
        };
        const std::multiset<int> m(ar, ar+sizeof(ar)/sizeof(ar[0]));
        assert(std::distance(m.begin(), m.end()) == m.size());
        assert(std::distance(m.cbegin(), m.cend()) == m.size());
        assert(std::distance(m.rbegin(), m.rend()) == m.size());
        assert(std::distance(m.crbegin(), m.crend()) == m.size());
        std::multiset<int, double>::const_iterator i;
        i = m.begin();
        for (int j = 1; j <= 8; ++j)
            for (int k = 0; k < 3; ++k, ++i)
                assert(*i == j);
    }
}
