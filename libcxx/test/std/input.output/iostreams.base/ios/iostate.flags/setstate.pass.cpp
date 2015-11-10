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

// void setstate(iostate state);

#include <ios>
#include <streambuf>
#include <cassert>

struct testbuf : public std::streambuf {};

int main()
{
    {
        std::ios ios(0);
        ios.setstate(std::ios::goodbit);
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
            ios.setstate(std::ios::goodbit);
            assert(false);
        }
        catch (std::ios::failure&)
        {
            assert(ios.rdstate() == std::ios::badbit);
        }
        try
        {
            ios.setstate(std::ios::eofbit);
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
        ios.setstate(std::ios::goodbit);
        assert(ios.rdstate() == std::ios::goodbit);
        ios.setstate(std::ios::eofbit);
        assert(ios.rdstate() == std::ios::eofbit);
        ios.setstate(std::ios::failbit);
        assert(ios.rdstate() == (std::ios::eofbit | std::ios::failbit));
    }
}
