//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <typeindex>

// class type_index

// template <>
// struct hash<type_index>
// {
//     size_t operator()(type_index index) const;
// };

// UNSUPPORTED: no-rtti

#include <typeindex>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
#if TEST_STD_VER <= 14
    typedef std::hash<std::type_index> H;
    static_assert((std::is_same<typename H::argument_type, std::type_index>::value), "" );
    static_assert((std::is_same<typename H::result_type, std::size_t>::value), "" );
#endif

    std::type_index t1 = typeid(int);
    assert(std::hash<std::type_index>()(t1) == t1.hash_code());

  return 0;
}
