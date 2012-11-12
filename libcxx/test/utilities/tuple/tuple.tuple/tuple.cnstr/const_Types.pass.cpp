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

// explicit tuple(const T&...);

#include <tuple>
#include <string>
#include <cassert>

int main()
{
    {
        std::tuple<int> t(2);
        assert(std::get<0>(t) == 2);
    }
    {
        std::tuple<int, char*> t(2, 0);
        assert(std::get<0>(t) == 2);
        assert(std::get<1>(t) == nullptr);
    }
    {
        std::tuple<int, char*> t(2, nullptr);
        assert(std::get<0>(t) == 2);
        assert(std::get<1>(t) == nullptr);
    }
    {
        std::tuple<int, char*, std::string> t(2, nullptr, "text");
        assert(std::get<0>(t) == 2);
        assert(std::get<1>(t) == nullptr);
        assert(std::get<2>(t) == "text");
    }
    // extensions
    {
        std::tuple<int, char*, std::string> t(2);
        assert(std::get<0>(t) == 2);
        assert(std::get<1>(t) == nullptr);
        assert(std::get<2>(t) == "");
    }
    {
        std::tuple<int, char*, std::string> t(2, nullptr);
        assert(std::get<0>(t) == 2);
        assert(std::get<1>(t) == nullptr);
        assert(std::get<2>(t) == "");
    }
    {
        std::tuple<int, char*, std::string, double> t(2, nullptr, "text");
        assert(std::get<0>(t) == 2);
        assert(std::get<1>(t) == nullptr);
        assert(std::get<2>(t) == "text");
        assert(std::get<3>(t) == 0.0);
    }
}
