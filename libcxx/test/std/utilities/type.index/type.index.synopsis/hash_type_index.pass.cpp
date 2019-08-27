//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

#include "test_macros.h"
#if TEST_STD_VER >= 11
#include "poisoned_hash_helper.h"
#endif

int main(int, char**)
{
  {
    typedef std::hash<std::type_index> H;
    static_assert((std::is_same<typename H::argument_type, std::type_index>::value), "" );
    static_assert((std::is_same<typename H::result_type, std::size_t>::value), "" );
  }
#if TEST_STD_VER >= 11
  {
    test_hash_enabled_for_type<std::type_index>(std::type_index(typeid(int)));
  }
#endif

  return 0;
}
