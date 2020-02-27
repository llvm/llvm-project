//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: libcpp-has-no-threads

// <atomic>

// Test nested types

// template <class T>
// class atomic
// {
// public:
//     typedef T                                        value_type;
// };

#include <atomic>
#include <type_traits>

#include <thread>
#include <chrono>
#if TEST_STD_VER >= 20
# include <memory>
#endif

#include "test_macros.h"

template <class A>
void
test_atomic()
{
    A a; (void)a;
#if TEST_STD_VER >= 17
    static_assert((std::is_same<typename A::value_type, decltype(a.load())>::value), "");
#endif
}

template <class T>
void
test()
{
    using A = std::atomic<T>;
#if TEST_STD_VER >= 17
    static_assert((std::is_same<typename A::value_type, T>::value), "");
#endif
    test_atomic<A>();
}

struct TriviallyCopyable {
    int i_;
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
    test<char16_t>           ();
    test<char32_t>           ();
    test<wchar_t>            ();

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
    test<std::thread::id>();
    test<std::chrono::nanoseconds>();
    test<float>();

#if TEST_STD_VER >= 20
    test_atomic<std::atomic_signed_lock_free>();
    test_atomic<std::atomic_unsigned_lock_free>();
/*
    test<std::shared_ptr<int>>();
*/
#endif

    return 0;
}
