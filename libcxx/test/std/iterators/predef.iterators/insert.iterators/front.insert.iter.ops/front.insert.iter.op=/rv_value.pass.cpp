//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <iterator>

// front_insert_iterator

// front_insert_iterator<Cont>&
//   operator=(Cont::value_type&& value);

#include <cassert>
#include <iterator>
#include <list>
#include <memory>

#include "test_macros.h"
#include "test_constexpr_container.h"

template <class C>
TEST_CONSTEXPR_CXX20 bool
test(C c)
{
    std::front_insert_iterator<C> i(c);
    i = typename C::value_type();
    assert(c.front() == typename C::value_type());
    return true;
}

int main(int, char**)
{
    test(std::list<std::unique_ptr<int> >());
#if TEST_STD_VER >= 20
    test(ConstexprFixedCapacityDeque<int, 10>());
    static_assert(test(ConstexprFixedCapacityDeque<int, 10>()));
#endif
    return 0;
}
