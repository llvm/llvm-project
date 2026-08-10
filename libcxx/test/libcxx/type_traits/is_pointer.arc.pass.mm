//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03
// REQUIRES: has-fobjc-arc
// ADDITIONAL_COMPILE_FLAGS: -fobjc-arc

// <type_traits>

// std::is_pointer

// Test that we correctly handle Objective-C++ ARC qualifiers on pointers.

#include <type_traits>
#include "test_macros.h"


template <typename T>
void assert_is_pointer() {
    static_assert(std::is_pointer<T>::value, "");
#if TEST_STD_VER > 14
    static_assert(std::is_pointer_v<T>, "");
#endif
}

template <typename T>
void test_is_pointer() {
    assert_is_pointer<T>();

    assert_is_pointer<T __weak>();
    assert_is_pointer<T __strong>();
    assert_is_pointer<T __autoreleasing>();
    assert_is_pointer<T __unsafe_unretained>();

    assert_is_pointer<T __weak const>();
    assert_is_pointer<T __strong const>();
    assert_is_pointer<T __autoreleasing const>();
    assert_is_pointer<T __unsafe_unretained const>();

    assert_is_pointer<T __weak volatile>();
    assert_is_pointer<T __strong volatile>();
    assert_is_pointer<T __autoreleasing volatile>();
    assert_is_pointer<T __unsafe_unretained volatile>();

    assert_is_pointer<T __weak const volatile>();
    assert_is_pointer<T __strong const volatile>();
    assert_is_pointer<T __autoreleasing const volatile>();
    assert_is_pointer<T __unsafe_unretained const volatile>();
}

@class Foo;

int main(int, char**) {
    test_is_pointer<id>();
    test_is_pointer<id const>();
    test_is_pointer<id volatile>();
    test_is_pointer<id const volatile>();

    test_is_pointer<Foo*>();
    test_is_pointer<Foo const*>();
    test_is_pointer<Foo volatile*>();
    test_is_pointer<Foo const volatile*>();

    test_is_pointer<void*>();
    test_is_pointer<void const*>();
    test_is_pointer<void volatile*>();
    test_is_pointer<void const volatile*>();

    return 0;
}
