//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// is_array

#include <type_traits>
#include <cstddef>        // for std::nullptr_t
#include "test_macros.h"

template <class T>
void test_is_array()
{
    static_assert( std::is_array<T>::value, "");
    static_assert( std::is_array<const T>::value, "");
    static_assert( std::is_array<volatile T>::value, "");
    static_assert( std::is_array<const volatile T>::value, "");
#if TEST_STD_VER > 14
    static_assert( std::is_array_v<T>, "");
    static_assert( std::is_array_v<const T>, "");
    static_assert( std::is_array_v<volatile T>, "");
    static_assert( std::is_array_v<const volatile T>, "");
#endif
}

template <class T>
void test_is_not_array()
{
    static_assert(!std::is_array<T>::value, "");
    static_assert(!std::is_array<const T>::value, "");
    static_assert(!std::is_array<volatile T>::value, "");
    static_assert(!std::is_array<const volatile T>::value, "");
#if TEST_STD_VER > 14
    static_assert(!std::is_array_v<T>, "");
    static_assert(!std::is_array_v<const T>, "");
    static_assert(!std::is_array_v<volatile T>, "");
    static_assert(!std::is_array_v<const volatile T>, "");
#endif
}

class Empty
{
};

class NotEmpty
{
    virtual ~NotEmpty();
};

union Union {};

struct bit_zero
{
    int :  0;
};

class Abstract
{
    virtual ~Abstract() = 0;
};

enum Enum {zero, one};
struct incomplete_type;

typedef void (*FunctionPtr)();

int main(int, char**)
{
    test_is_array<char[3]>();
    // Android clang-r536225 identifies as clang-19.0, but it predates the
    // LLVM 19.0.0 release. It lacks llvm.org/pr86652, which changed __is_array
    // to return false for T[0]. llvm.org/pr93037 relies on that change for
    // correct handling of std::is_array<T[0]>. This test will pass as long as
    // Clang and libc++ come from the same LLVM commit, but we can't detect that
    // here.
#if !defined(__ANDROID__) || TEST_CLANG_VER != 1900
    test_is_not_array<char[0]>();
#endif
    test_is_array<char[]>();
    test_is_array<Union[]>();

    test_is_not_array<std::nullptr_t>();
    test_is_not_array<void>();
    test_is_not_array<int&>();
    test_is_not_array<int&&>();
    test_is_not_array<int*>();
    test_is_not_array<double>();
    test_is_not_array<const int*>();
    test_is_not_array<Enum>();
    test_is_not_array<Union>();
    test_is_not_array<FunctionPtr>();
    test_is_not_array<Empty>();
    test_is_not_array<bit_zero>();
    test_is_not_array<NotEmpty>();
    test_is_not_array<incomplete_type>();  //  LWG#2582

  return 0;
}
