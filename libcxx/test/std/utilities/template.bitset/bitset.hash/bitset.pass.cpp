//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <functional>

// template <class T>
// struct hash
//     : public unary_function<T, size_t>
// {
//     size_t operator()(T val) const;
// };

#include <bitset>
#include <cassert>
#include <type_traits>

#include "test_macros.h"

template <std::size_t N>
void
test()
{
    typedef std::bitset<N> T;
    typedef std::hash<T> H;
    static_assert((std::is_same<typename H::argument_type, T>::value), "" );
    static_assert((std::is_same<typename H::result_type, std::size_t>::value), "" );
    ASSERT_NOEXCEPT(H()(T()));

    H h;
    T bs(static_cast<unsigned long long>(N));
    const std::size_t result = h(bs);
    LIBCPP_ASSERT(result == N);
    ((void)result); // Prevent unused warning
}

int main()
{
    test<0>();
    test<10>();
    test<100>();
    test<1000>();
}
