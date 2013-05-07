//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <valarray>

// template<class T> class valarray;

// valarray& operator=(valarray&& v);

#include <valarray>
#include <cassert>

int main()
{
#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES
    {
        typedef int T;
        T a[] = {1, 2, 3, 4, 5};
        const unsigned N = sizeof(a)/sizeof(a[0]);
        std::valarray<T> v(a, N);
        std::valarray<T> v2;
        v2 = std::move(v);
        assert(v2.size() == N);
        assert(v.size() == 0);
        for (int i = 0; i < v2.size(); ++i)
            assert(v2[i] == a[i]);
    }
    {
        typedef double T;
        T a[] = {1, 2.5, 3, 4.25, 5};
        const unsigned N = sizeof(a)/sizeof(a[0]);
        std::valarray<T> v(a, N);
        std::valarray<T> v2;
        v2 = std::move(v);
        assert(v2.size() == N);
        assert(v.size() == 0);
        for (int i = 0; i < v2.size(); ++i)
            assert(v2[i] == a[i]);
    }
    {
        typedef std::valarray<double> T;
        T a[] = {T(1), T(2), T(3), T(4), T(5)};
        const unsigned N = sizeof(a)/sizeof(a[0]);
        std::valarray<T> v(a, N);
        std::valarray<T> v2(a, N-2);
        v2 = std::move(v);
        assert(v2.size() == N);
        assert(v.size() == 0);
        for (int i = 0; i < N; ++i)
        {
            assert(v2[i].size() == a[i].size());
            for (int j = 0; j < a[i].size(); ++j)
                assert(v2[i][j] == a[i][j]);
        }
    }
#endif  // _LIBCPP_HAS_NO_RVALUE_REFERENCES
}
