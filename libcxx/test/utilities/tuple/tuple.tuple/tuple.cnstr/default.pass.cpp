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

// constexpr tuple();

#include <tuple>
#include <string>
#include <cassert>

#include "../DefaultOnly.h"

int main()
{
    {
        std::tuple<> t;
    }
    {
        std::tuple<int> t;
        assert(std::get<0>(t) == 0);
    }
    {
        std::tuple<int, char*> t;
        assert(std::get<0>(t) == 0);
        assert(std::get<1>(t) == nullptr);
    }
    {
        std::tuple<int, char*, std::string> t;
        assert(std::get<0>(t) == 0);
        assert(std::get<1>(t) == nullptr);
        assert(std::get<2>(t) == "");
    }
    {
        std::tuple<int, char*, std::string, DefaultOnly> t;
        assert(std::get<0>(t) == 0);
        assert(std::get<1>(t) == nullptr);
        assert(std::get<2>(t) == "");
        assert(std::get<3>(t) == DefaultOnly());
    }
#ifndef _LIBCPP_HAS_NO_CONSTEXPR
    {
        constexpr std::tuple<> t;
    }
    {
        constexpr std::tuple<int> t;
        assert(std::get<0>(t) == 0);
    }
    {
        constexpr std::tuple<int, char*> t;
        assert(std::get<0>(t) == 0);
        assert(std::get<1>(t) == nullptr);
    }
#endif
}
