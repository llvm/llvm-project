//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_HASH_H
#define TEST_HASH_H

#include <cstddef>
#include <type_traits>

template <class C>
class test_hash
    : private C
{
    int data_;
public:
    explicit test_hash(int data = 0) : data_(data) {}

    std::size_t
    operator()(typename std::add_lvalue_reference<const typename C::argument_type>::type x) const
        {return C::operator()(x);}

    bool operator==(const test_hash& c) const
        {return data_ == c.data_;}
};

#endif // TEST_HASH_H
