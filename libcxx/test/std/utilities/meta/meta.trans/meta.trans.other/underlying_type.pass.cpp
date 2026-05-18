//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// underlying_type
//  As of C++20, std::underlying_type is SFINAE-friendly; if you hand it
//  a non-enumeration, it returns an empty struct.

#include <type_traits>
#include <climits>
#include <cstddef>

#include "test_macros.h"


//  MSVC's ABI doesn't follow the standard
#if !defined(_WIN32) || defined(__MINGW32__)
    #define TEST_UNSIGNED_UNDERLYING_TYPE 1
#endif


#if TEST_STD_VER > 17
template <class, class = std::void_t<>>
struct has_type_member : std::false_type {};

template <class T>
struct has_type_member<T,
           std::void_t<typename std::underlying_type<T>::type>> : std::true_type {};

struct S {};
union U { int i; float f;};
#endif

template <typename T, typename Expected>
void check()
{
    ASSERT_SAME_TYPE(Expected, typename std::underlying_type<T>::type);
#if TEST_STD_VER > 11
    ASSERT_SAME_TYPE(Expected, typename std::underlying_type_t<T>);
#endif
}

enum E { V = INT_MIN };

#ifdef TEST_UNSIGNED_UNDERLYING_TYPE
enum F { W = UINT_MAX };
#endif // TEST_UNSIGNED_UNDERLYING_TYPE

#if TEST_STD_VER >= 11
enum G : char {};
enum class H { red, green = 20, blue };
enum class I : long { red, green = 20, blue };
enum struct J { red, green = 20, blue };
enum struct K : short { red, green = 20, blue };
#endif

int main(int, char**)
{
//  Basic tests
    check<E, int>();
#ifdef TEST_UNSIGNED_UNDERLYING_TYPE
    check<F, unsigned>();
#endif // TEST_UNSIGNED_UNDERLYING_TYPE

//  Class enums and enums with specified underlying type
#if TEST_STD_VER >= 11
    check<G, char>();
    check<H, int>();
    check<I, long>();
    check<J, int>();
    check<K, short>();
#endif

//  SFINAE-able underlying_type
#if TEST_STD_VER > 17
    static_assert( has_type_member<E>::value, "");
#ifdef TEST_UNSIGNED_UNDERLYING_TYPE
    static_assert( has_type_member<F>::value, "");
#endif // TEST_UNSIGNED_UNDERLYING_TYPE
    static_assert( has_type_member<G>::value, "");

    static_assert(!has_type_member<void>::value, "");
    static_assert(!has_type_member<int>::value, "");
    static_assert(!has_type_member<double>::value, "");
    static_assert(!has_type_member<int[]>::value, "");
    static_assert(!has_type_member<S>::value, "");
    static_assert(!has_type_member<void (S::*)(int)>::value, "");
    static_assert(!has_type_member<void (S::*)(int, ...)>::value, "");
    static_assert(!has_type_member<U>::value, "");
    static_assert(!has_type_member<void(int)>::value, "");
    static_assert(!has_type_member<void(int, ...)>::value, "");
    static_assert(!has_type_member<int&>::value, "");
    static_assert(!has_type_member<int&&>::value, "");
    static_assert(!has_type_member<int*>::value, "");
    static_assert(!has_type_member<std::nullptr_t>::value, "");
#endif

  return 0;
}
