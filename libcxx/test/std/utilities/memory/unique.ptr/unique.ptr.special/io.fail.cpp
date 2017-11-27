//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <memory>

// unique_ptr

// template<class CharT, class Traits, class Y, class D>
//   basic_ostream<CharT, Traits>&
//   operator<<(basic_ostream<CharT, Traits>& os, const unique_ptr<Y, D>& p);

//   -?- Remarks: This function shall not participate in overload resolution unless os << p.get() is a valid expression.

#include <memory>
#include <sstream>
#include <cassert>

class A {};

int main()
{
    std::unique_ptr<A> p(new A);
    std::ostringstream os;
    os << p;
}
