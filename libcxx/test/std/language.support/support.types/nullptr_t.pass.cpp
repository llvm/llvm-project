//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cstddef>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

// typedef decltype(nullptr) nullptr_t;

struct A
{
    A(std::nullptr_t) {}
};

template <class T>
void test_conversions()
{
    {
        T p = 0;
        assert(p == nullptr);
    }
    {
        T p = nullptr;
        assert(p == nullptr);
        assert(nullptr == p);
        assert(!(p != nullptr));
        assert(!(nullptr != p));
    }
}

template <class T> struct Voider { typedef void type; };
template <class T, class = void> struct has_less : std::false_type {};

template <class T> struct has_less<T,
    typename Voider<decltype(std::declval<T>() < nullptr)>::type> : std::true_type {};

template <class T>
void test_comparisons()
{
    T p = nullptr;
    assert(p == nullptr);
    assert(!(p != nullptr));
    assert(nullptr == p);
    assert(!(nullptr != p));
}

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnull-conversion"
#endif
void test_nullptr_conversions() {
    {
        bool b(nullptr);
        assert(!b);
    }
}
#if defined(__clang__)
#pragma clang diagnostic pop
#endif


int main(int, char**)
{
    static_assert(sizeof(std::nullptr_t) == sizeof(void*),
                  "sizeof(std::nullptr_t) == sizeof(void*)");

    {
        test_conversions<std::nullptr_t>();
        test_conversions<void*>();
        test_conversions<A*>();
        test_conversions<void(*)()>();
        test_conversions<void(A::*)()>();
        test_conversions<int A::*>();
    }
    {
        // TODO: Enable this assertion when GCC compilers implements http://wg21.link/CWG583.
#if !defined(TEST_COMPILER_GCC)
        static_assert(!has_less<std::nullptr_t>::value, "");
#endif
        test_comparisons<std::nullptr_t>();
        test_comparisons<void*>();
        test_comparisons<A*>();
        test_comparisons<void(*)()>();
    }
    test_nullptr_conversions();

  return 0;
}
