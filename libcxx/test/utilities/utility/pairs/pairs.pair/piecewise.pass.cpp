//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <utility>

// template <class T1, class T2> struct pair

// template <class... Args1, class... Args2>
//     pair(piecewise_construct_t, tuple<Args1...> first_args,
//                                 tuple<Args2...> second_args);

#include <utility>
#include <tuple>
#include <cassert>

int main()
{
#ifndef _LIBCPP_HAS_NO_VARIADICS
    {
        typedef std::pair<int, int*> P1;
        typedef std::pair<int*, int> P2;
        typedef std::pair<P1, P2> P3;
        P3 p3(std::piecewise_construct, std::tuple<int, int*>(3, nullptr),
                                        std::tuple<int*, int>(nullptr, 4));
        assert(p3.first == P1(3, nullptr));
        assert(p3.second == P2(nullptr, 4));
    }
#endif  // _LIBCPP_HAS_NO_VARIADICS
}
