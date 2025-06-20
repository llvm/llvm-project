//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// XFAIL: FROZEN-CXX03-HEADERS-FIXME

#include <__tree>
#include <map>
#include <set>
#include <type_traits>

#include "test_macros.h"
#include "min_allocator.h"

void testKeyValueTrait() {
  {
    typedef int Tp;
    typedef std::__tree_key_value_types<Tp> Traits;
    static_assert((std::is_same<Traits::key_type, int>::value), "");
    static_assert((std::is_same<Traits::__container_value_type, Tp>::value), "");
    static_assert(Traits::__is_map == false, "");
  }
  {
    typedef std::pair<int, int> Tp;
    typedef std::__tree_key_value_types<Tp> Traits;
    static_assert((std::is_same<Traits::key_type, Tp>::value), "");
    static_assert((std::is_same<Traits::__container_value_type, Tp>::value), "");
    static_assert(Traits::__is_map == false, "");
  }
  {
    typedef std::pair<const int, int> Tp;
    typedef std::__tree_key_value_types<Tp> Traits;
    static_assert((std::is_same<Traits::key_type, Tp>::value), "");
    static_assert((std::is_same<Traits::__container_value_type, Tp>::value), "");
    static_assert(Traits::__is_map == false, "");
  }
  {
    typedef std::__value_type<int, int> Tp;
    typedef std::__tree_key_value_types<Tp> Traits;
    static_assert((std::is_same<Traits::key_type, int>::value), "");
    static_assert((std::is_same<Traits::mapped_type, int>::value), "");
    static_assert((std::is_same<Traits::__container_value_type, std::pair<const int, int> >::value), "");
    static_assert((std::is_same<Traits::__map_value_type, std::pair<const int, int> >::value), "");
    static_assert(Traits::__is_map == true, "");
  }
}

int main(int, char**) {
  testKeyValueTrait();

  return 0;
}
