//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <typeindex>

// struct hash<type_index>
//     : public unary_function<type_index, size_t>
// {
//     size_t operator()(type_index index) const;
// };

#include <typeindex>
#include <type_traits>

int main()
{
    static_assert((std::is_base_of<std::unary_function<std::type_index, std::size_t>,
                                   std::hash<std::type_index> >::value), "");
}
