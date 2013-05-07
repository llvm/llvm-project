//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// template<class E> class initializer_list;

// initializer_list();

#include <initializer_list>
#include <cassert>

struct A {};

int main()
{
#ifndef _LIBCPP_HAS_NO_GENERALIZED_INITIALIZERS
    std::initializer_list<A> il;
    assert(il.size() == 0);
#endif  // _LIBCPP_HAS_NO_GENERALIZED_INITIALIZERS
}
