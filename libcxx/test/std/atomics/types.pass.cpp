//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <atomic>

// Test nested types

// template <class T>
// class atomic
// {
// public:
//     typedef T value_type;
// };

#include <atomic>
#include <chrono>
#include <memory>
#include <type_traits>

#ifndef _LIBCPP_HAS_NO_THREADS
#   include <thread>
#endif

#include "test_macros.h"

template <class A, bool Integral>
struct test_atomic
{
    test_atomic()
    {
        A a; (void)a;
#if TEST_STD_VER >= 17
    static_assert((std::is_same_v<typename A::value_type, decltype(a.load())>), "");
#endif
    }
};

template <class A>
struct test_atomic<A, true>
{
    test_atomic()
    {
        A a; (void)a;
#if TEST_STD_VER >= 17
    static_assert((std::is_same_v<typename A::value_type, decltype(a.load())>), "");
    static_assert((std::is_same_v<typename A::value_type, typename A::difference_type>), "");
#endif
    }
};

template <class A>
struct test_atomic<A*, false>
{
    test_atomic()
    {
        A a; (void)a;
#if TEST_STD_VER >= 17
    static_assert((std::is_same_v<typename A::value_type, decltype(a.load())>), "");
    static_assert((std::is_same_v<typename A::difference_type, ptrdiff_t>), "");
#endif
    }
};

template <class T>
void
test()
{
    using A = std::atomic<T>;
#if TEST_STD_VER >= 17
    static_assert((std::is_same_v<typename A::value_type, T>), "");
#endif
    test_atomic<A, std::is_integral<T>::value && !std::is_same<T, bool>::value>();
}

struct TriviallyCopyable {
    int i_;
};

struct WeirdTriviallyCopyable
{
    char i, j, k; /* the 3 chars of doom */
};

struct PaddedTriviallyCopyable
{
    char i; int j; /* probably lock-free? */
};

struct LargeTriviallyCopyable
{
    int i, j[127]; /* decidedly not lock-free */
};

int main(int, char**)
{
    test<bool>               ();
    test<char>               ();
    test<signed char>        ();
    test<unsigned char>      ();
    test<short>              ();
    test<unsigned short>     ();
    test<int>                ();
    test<unsigned int>       ();
    test<long>               ();
    test<unsigned long>      ();
    test<long long>          ();
    test<unsigned long long> ();
#if TEST_STD_VER > 17 && defined(__cpp_char8_t)
    test<char8_t>            ();
#endif
    test<char16_t>           ();
    test<char32_t>           ();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    test<wchar_t>            ();
#endif

    test<int_least8_t>   ();
    test<uint_least8_t>  ();
    test<int_least16_t>  ();
    test<uint_least16_t> ();
    test<int_least32_t>  ();
    test<uint_least32_t> ();
    test<int_least64_t>  ();
    test<uint_least64_t> ();

    test<int_fast8_t>   ();
    test<uint_fast8_t>  ();
    test<int_fast16_t>  ();
    test<uint_fast16_t> ();
    test<int_fast32_t>  ();
    test<uint_fast32_t> ();
    test<int_fast64_t>  ();
    test<uint_fast64_t> ();

    test< int8_t>  ();
    test<uint8_t>  ();
    test< int16_t> ();
    test<uint16_t> ();
    test< int32_t> ();
    test<uint32_t> ();
    test< int64_t> ();
    test<uint64_t> ();

    test<intptr_t>  ();
    test<uintptr_t> ();
    test<size_t>    ();
    test<ptrdiff_t> ();
    test<intmax_t>  ();
    test<uintmax_t> ();

    test<uintmax_t> ();
    test<uintmax_t> ();

    test<TriviallyCopyable>();
    test<PaddedTriviallyCopyable>();
#ifndef __APPLE__ // Apple doesn't ship libatomic
    /*
        These aren't going to be lock-free,
        so some libatomic.a is necessary.
    */
    test<WeirdTriviallyCopyable>();
    test<LargeTriviallyCopyable>();
#endif

#ifndef _LIBCPP_HAS_NO_THREADS
    test<std::thread::id>();
#endif
    test<std::chrono::nanoseconds>();
    test<float>();

#if TEST_STD_VER >= 20
    test<std::atomic_signed_lock_free::value_type>();
    test<std::atomic_unsigned_lock_free::value_type>();
/*
    test<std::shared_ptr<int>>();
*/
#endif

    return 0;
}
