//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: c++98, c++03

// <type_traits>

// std::is_pointer

// Test that we correctly handle Objective-C++ ARC qualifiers on pointers.

#include <type_traits>


template <typename T>
void test_is_pointer() {
    static_assert(std::is_pointer<T>::value, "");

    static_assert(std::is_pointer<T __weak>::value, "");
    static_assert(std::is_pointer<T __strong>::value, "");
    static_assert(std::is_pointer<T __autoreleasing>::value, "");
    static_assert(std::is_pointer<T __unsafe_unretained>::value, "");

    static_assert(std::is_pointer<T __weak const>::value, "");
    static_assert(std::is_pointer<T __strong const>::value, "");
    static_assert(std::is_pointer<T __autoreleasing const>::value, "");
    static_assert(std::is_pointer<T __unsafe_unretained const>::value, "");

    static_assert(std::is_pointer<T __weak volatile>::value, "");
    static_assert(std::is_pointer<T __strong volatile>::value, "");
    static_assert(std::is_pointer<T __autoreleasing volatile>::value, "");
    static_assert(std::is_pointer<T __unsafe_unretained volatile>::value, "");

    static_assert(std::is_pointer<T __weak const volatile>::value, "");
    static_assert(std::is_pointer<T __strong const volatile>::value, "");
    static_assert(std::is_pointer<T __autoreleasing const volatile>::value, "");
    static_assert(std::is_pointer<T __unsafe_unretained const volatile>::value, "");
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
