//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <ios>

// template <class charT, class traits> class basic_ios

// void set_rdbuf(basic_streambuf<charT, traits>* sb);

#include <ios>
#include <streambuf>
#include <cassert>

struct testbuf
    : public std::streambuf
{
};

struct testios
    : public std::ios
{
    testios(std::streambuf* p) : std::ios(p) {}
    void set_rdbuf(std::streambuf* x) {std::ios::set_rdbuf(x);}
};

int main()
{
    testbuf sb1;
    testbuf sb2;
    testios ios(&sb1);
    try
    {
        ios.setstate(std::ios::badbit);
        ios.exceptions(std::ios::badbit);
    }
    catch (...)
    {
    }
    ios.set_rdbuf(&sb2);
    assert(ios.rdbuf() == &sb2);
    try
    {
        ios.setstate(std::ios::badbit);
        ios.exceptions(std::ios::badbit);
    }
    catch (...)
    {
    }
    ios.set_rdbuf(0);
    assert(ios.rdbuf() == 0);
}
