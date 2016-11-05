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

// template<class... Types>
//   tuple<Types&...> tie(Types&... t);

// UNSUPPORTED: c++98, c++03

#include <tuple>
#include <string>
#include <cassert>

#include "test_macros.h"

int main()
{
    {
        int i = 0;
        std::string s;
        std::tie(i, std::ignore, s) = std::make_tuple(42, 3.14, "C++");
        assert(i == 42);
        assert(s == "C++");
    }
#if TEST_STD_VER > 11
    {
        static constexpr int i = 42;
        static constexpr double f = 1.1;
        constexpr std::tuple<const int &, const double &> t = std::tie(i, f);
        static_assert ( std::get<0>(t) == 42, "" );
        static_assert ( std::get<1>(t) == 1.1, "" );
    }
#endif
}
