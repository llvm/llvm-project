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

// template <>
// struct hash<type_index>
//     : public unary_function<type_index, size_t>
// {
//     size_t operator()(type_index index) const;
// };

#include <typeindex>
#include <cassert>

int main()
{
    std::type_index t1 = typeid(int);
    assert(std::hash<std::type_index>()(t1) == t1.hash_code());
}
