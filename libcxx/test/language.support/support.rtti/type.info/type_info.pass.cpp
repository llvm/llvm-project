//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// test type_info

#include <typeinfo>
#include <cstring>
#include <cassert>

int main()
{
    const std::type_info& t1 = typeid(int);
    const std::type_info& t2 = typeid(int);
    assert(t1 == t2);
    const std::type_info& t3 = typeid(short);
    assert(t1 != t3);
    assert(!t1.before(t2));
    assert(strcmp(t1.name(), t2.name()) == 0);
    assert(strcmp(t1.name(), t3.name()) != 0);
}
