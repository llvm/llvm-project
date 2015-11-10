//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// XFAIL: libcpp-no-exceptions
// <ios>

// template <class charT, class traits> class basic_ios

// void clear(iostate state = goodbit);

#include <ios>
#include <streambuf>
#include <cassert>

struct testbuf : public std::streambuf {};

int main()
{
    {
        std::ios ios(0);
        ios.clear();
        assert(ios.rdstate() == std::ios::badbit);
        try
        {
            ios.exceptions(std::ios::badbit);
        }
        catch (...)
        {
        }
        try
        {
            ios.clear();
            assert(false);
        }
        catch (std::ios::failure&)
        {
            assert(ios.rdstate() == std::ios::badbit);
        }
        try
        {
            ios.clear(std::ios::eofbit);
            assert(false);
        }
        catch (std::ios::failure&)
        {
            assert(ios.rdstate() == (std::ios::eofbit | std::ios::badbit));
        }
    }
    {
        testbuf sb;
        std::ios ios(&sb);
        ios.clear();
        assert(ios.rdstate() == std::ios::goodbit);
        ios.exceptions(std::ios::badbit);
        ios.clear();
        assert(ios.rdstate() == std::ios::goodbit);
        ios.clear(std::ios::eofbit);
        assert(ios.rdstate() == std::ios::eofbit);
    }
}
