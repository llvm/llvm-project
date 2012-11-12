//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <tuple>

// template <class... Types> class tuple;

// template <class U1, class U2>
//   tuple& operator=(const pair<U1, U2>& u);

#include <tuple>
#include <utility>
#include <cassert>

int main()
{
    {
        typedef std::pair<double, char> T0;
        typedef std::tuple<int, short> T1;
        T0 t0(2.5, 'a');
        T1 t1;
        t1 = t0;
        assert(std::get<0>(t1) == 2);
        assert(std::get<1>(t1) == short('a'));
    }
}
