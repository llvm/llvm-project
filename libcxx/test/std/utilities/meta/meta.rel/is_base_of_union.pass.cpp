//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// is_base_of

#include <type_traits>

#include "test_macros.h"

//  Clang before v9 and apple-clang up to and including v11 do not
//  report that unions are never base classes - nor can they have bases.
//  See https://reviews.llvm.org/D61858
// XFAIL: apple-clang-6.0, apple-clang-7.0, apple-clang-8.0
// XFAIL: apple-clang-9.0, apple-clang-9.1, apple-clang-10.0, apple-clang-11.0.0
// XFAIL: clang-3.3, clang-3.4, clang-3.5, clang-3.6, clang-3.7, clang-3.8, clang-3.9
// XFAIL: clang-4.0, clang-5.0, clang-6.0, clang-7.0, clang-8.0


template <class T, class U>
void test_is_base_of()
{
    static_assert((std::is_base_of<T, U>::value), "");
    static_assert((std::is_base_of<const T, U>::value), "");
    static_assert((std::is_base_of<T, const U>::value), "");
    static_assert((std::is_base_of<const T, const U>::value), "");
#if TEST_STD_VER > 14
    static_assert((std::is_base_of_v<T, U>), "");
    static_assert((std::is_base_of_v<const T, U>), "");
    static_assert((std::is_base_of_v<T, const U>), "");
    static_assert((std::is_base_of_v<const T, const U>), "");
#endif
}

template <class T, class U>
void test_is_not_base_of()
{
    static_assert((!std::is_base_of<T, U>::value), "");
}

struct B {};
struct B1 : B {};
struct B2 : B {};
struct D : private B1, private B2 {};
union U0;
union U1 {};
struct I0;
struct I1 {};

int main(int, char**)
{
    // A union is never the base class of anything (including incomplete types)
    test_is_not_base_of<U0, B>();
    test_is_not_base_of<U0, B1>();
    test_is_not_base_of<U0, B2>();
    test_is_not_base_of<U0, D>();
    test_is_not_base_of<U1, B>();
    test_is_not_base_of<U1, B1>();
    test_is_not_base_of<U1, B2>();
    test_is_not_base_of<U1, D>();
    test_is_not_base_of<U0, I0>();
    test_is_not_base_of<U1, I1>();
    test_is_not_base_of<U0, U1>();
    test_is_not_base_of<U0, int>();
    test_is_not_base_of<U1, int>();
    test_is_not_base_of<I0, int>();
    test_is_not_base_of<I1, int>();

    // A union never has base classes (including incomplete types)
    test_is_not_base_of<B,  U0>();
    test_is_not_base_of<B1, U0>();
    test_is_not_base_of<B2, U0>();
    test_is_not_base_of<D,  U0>();
    test_is_not_base_of<B,  U1>();
    test_is_not_base_of<B1, U1>();
    test_is_not_base_of<B2, U1>();
    test_is_not_base_of<D,  U1>();
    test_is_not_base_of<I0, U0>();
    test_is_not_base_of<I1, U1>();
    test_is_not_base_of<U1, U0>();
    test_is_not_base_of<int, U0>();
    test_is_not_base_of<int, U1>();
    test_is_not_base_of<int, I0>();
    test_is_not_base_of<int, I1>();

  return 0;
}
