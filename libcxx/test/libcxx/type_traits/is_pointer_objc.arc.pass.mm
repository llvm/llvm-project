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
void test() {
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
    test<id>();
    test<id const>();
    test<id volatile>();
    test<id const volatile>();
    test<Foo*>();
    test<Foo const*>();
    test<Foo volatile*>();
    test<Foo const volatile*>();

    return 0;
}
