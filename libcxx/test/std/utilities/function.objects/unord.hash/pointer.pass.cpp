//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS

// <functional>

// template <class T>
// struct hash
//     : public unary_function<T, size_t>
// {
//     size_t operator()(T val) const;
// };

// Not very portable

#include <functional>
#include <cassert>
#include <type_traits>
#include <limits>

#include "test_macros.h"

template <class T>
void
test()
{
    typedef std::hash<T> H;
#if TEST_STD_VER <= 17
    static_assert((std::is_same<typename H::argument_type, T>::value), "");
    static_assert((std::is_same<typename H::result_type, std::size_t>::value), "");
#endif
    ASSERT_NOEXCEPT(H()(T()));
    H h;

    typedef typename std::remove_pointer<T>::type type;
    type i;
    type j;
    assert(h(&i) != h(&j));
}

// can't hash nullptr_t until C++17
void test_nullptr()
{
#if TEST_STD_VER > 14
    typedef std::nullptr_t T;
    typedef std::hash<T> H;
#if TEST_STD_VER <= 17
    static_assert((std::is_same<typename H::argument_type, T>::value), "" );
    static_assert((std::is_same<typename H::result_type, std::size_t>::value), "" );
#endif
    ASSERT_NOEXCEPT(H()(T()));
#endif
}

int main(int, char**)
{
    test<int*>();
    test_nullptr();

  return 0;
}
