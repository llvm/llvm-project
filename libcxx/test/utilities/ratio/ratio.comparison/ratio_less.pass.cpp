//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// test ratio_less

#include <ratio>

int main()
{
    {
    typedef std::ratio<1, 1> R1;
    typedef std::ratio<1, 1> R2;
    static_assert((!std::ratio_less<R1, R2>::value), "");
    }
    {
    typedef std::ratio<0x7FFFFFFFFFFFFFFFLL, 1> R1;
    typedef std::ratio<0x7FFFFFFFFFFFFFFFLL, 1> R2;
    static_assert((!std::ratio_less<R1, R2>::value), "");
    }
    {
    typedef std::ratio<-0x7FFFFFFFFFFFFFFFLL, 1> R1;
    typedef std::ratio<-0x7FFFFFFFFFFFFFFFLL, 1> R2;
    static_assert((!std::ratio_less<R1, R2>::value), "");
    }
    {
    typedef std::ratio<1, 0x7FFFFFFFFFFFFFFFLL> R1;
    typedef std::ratio<1, 0x7FFFFFFFFFFFFFFFLL> R2;
    static_assert((!std::ratio_less<R1, R2>::value), "");
    }
    {
    typedef std::ratio<1, 1> R1;
    typedef std::ratio<1, -1> R2;
    static_assert((!std::ratio_less<R1, R2>::value), "");
    }
    {
    typedef std::ratio<0x7FFFFFFFFFFFFFFFLL, 1> R1;
    typedef std::ratio<-0x7FFFFFFFFFFFFFFFLL, 1> R2;
    static_assert((!std::ratio_less<R1, R2>::value), "");
    }
    {
    typedef std::ratio<-0x7FFFFFFFFFFFFFFFLL, 1> R1;
    typedef std::ratio<0x7FFFFFFFFFFFFFFFLL, 1> R2;
    static_assert((std::ratio_less<R1, R2>::value), "");
    }
    {
    typedef std::ratio<1, 0x7FFFFFFFFFFFFFFFLL> R1;
    typedef std::ratio<1, -0x7FFFFFFFFFFFFFFFLL> R2;
    static_assert((!std::ratio_less<R1, R2>::value), "");
    }
    {
    typedef std::ratio<0x7FFFFFFFFFFFFFFFLL, 0x7FFFFFFFFFFFFFFELL> R1;
    typedef std::ratio<0x7FFFFFFFFFFFFFFDLL, 0x7FFFFFFFFFFFFFFCLL> R2;
    static_assert((std::ratio_less<R1, R2>::value), "");
    }
    {
    typedef std::ratio<0x7FFFFFFFFFFFFFFDLL, 0x7FFFFFFFFFFFFFFCLL> R1;
    typedef std::ratio<0x7FFFFFFFFFFFFFFFLL, 0x7FFFFFFFFFFFFFFELL> R2;
    static_assert((!std::ratio_less<R1, R2>::value), "");
    }
    {
    typedef std::ratio<-0x7FFFFFFFFFFFFFFDLL, 0x7FFFFFFFFFFFFFFCLL> R1;
    typedef std::ratio<-0x7FFFFFFFFFFFFFFFLL, 0x7FFFFFFFFFFFFFFELL> R2;
    static_assert((std::ratio_less<R1, R2>::value), "");
    }
    {
    typedef std::ratio<0x7FFFFFFFFFFFFFFFLL, 0x7FFFFFFFFFFFFFFELL> R1;
    typedef std::ratio<0x7FFFFFFFFFFFFFFELL, 0x7FFFFFFFFFFFFFFDLL> R2;
    static_assert((std::ratio_less<R1, R2>::value), "");
    }
    {
    typedef std::ratio<641981, 1339063> R1;
    typedef std::ratio<1291640, 2694141LL> R2;
    static_assert((!std::ratio_less<R1, R2>::value), "");
    }
    {
    typedef std::ratio<1291640, 2694141LL> R1;
    typedef std::ratio<641981, 1339063> R2;
    static_assert((std::ratio_less<R1, R2>::value), "");
    }
}
