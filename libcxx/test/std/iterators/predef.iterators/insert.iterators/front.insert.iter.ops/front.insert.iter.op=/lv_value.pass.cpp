//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// front_insert_iterator

// front_insert_iterator<Cont>&
//   operator=(const Cont::value_type& value);

#include <cassert>
#include <iterator>
#include <list>

#include "test_macros.h"
#include "nasty_containers.h"
#include "test_constexpr_container.h"

template <class C>
TEST_CONSTEXPR_CXX20 bool
test(C c)
{
    const typename C::value_type v = typename C::value_type();
    std::front_insert_iterator<C> i(c);
    i = v;
    assert(c.front() == v);
    return true;
}

class Copyable
{
    int data_;
public:
    Copyable() : data_(0) {}
    ~Copyable() {data_ = -1;}

    friend bool operator==(const Copyable& x, const Copyable& y)
        {return x.data_ == y.data_;}
};

int main(int, char**)
{
    test(std::list<Copyable>());
    test(nasty_list<Copyable>());
#if TEST_STD_VER >= 20
    test(ConstexprFixedCapacityDeque<int, 10>());
    static_assert(test(ConstexprFixedCapacityDeque<int, 10>()));
#endif
    return 0;
}
