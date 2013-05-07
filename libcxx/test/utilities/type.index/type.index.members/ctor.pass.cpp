//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <typeindex>

// class type_index

// type_index(const type_info& rhs);

#include <typeindex>
#include <cassert>

int main()
{
    std::type_index t1 = typeid(int);
}
