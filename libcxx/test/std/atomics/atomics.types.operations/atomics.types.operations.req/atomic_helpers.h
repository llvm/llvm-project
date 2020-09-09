//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ATOMIC_HELPERS_H
#define ATOMIC_HELPERS_H

#include <cassert>

#include "test_macros.h"

struct UserAtomicType
{
    int i;

    explicit UserAtomicType(int d = 0) TEST_NOEXCEPT : i(d) {}

    friend bool operator==(const UserAtomicType& x, const UserAtomicType& y)
    { return x.i == y.i; }
};

struct WeirdUserAtomicType
{
    char i, j, k; /* the 3 chars of doom */

    explicit WeirdUserAtomicType(int d = 0) TEST_NOEXCEPT : i(d) {}

    friend bool operator==(const WeirdUserAtomicType& x, const WeirdUserAtomicType& y)
    { return x.i == y.i; }
};

struct PaddedUserAtomicType
{
    char i; int j; /* probably lock-free? */

    explicit PaddedUserAtomicType(int d = 0) TEST_NOEXCEPT : i(d) {}

    friend bool operator==(const PaddedUserAtomicType& x, const PaddedUserAtomicType& y)
    { return x.i == y.i; }
};

struct LargeUserAtomicType
{
    int i, j[127]; /* decidedly not lock-free */

    LargeUserAtomicType(int d = 0) TEST_NOEXCEPT : i(d)
    {}

    friend bool operator==(const LargeUserAtomicType& x, const LargeUserAtomicType& y)
    { return x.i == y.i; }
};

template < template <class TestArg> class TestFunctor >
struct TestEachIntegralType {
    void operator()() const {
        TestFunctor<char>()();
        TestFunctor<signed char>()();
        TestFunctor<unsigned char>()();
        TestFunctor<short>()();
        TestFunctor<unsigned short>()();
        TestFunctor<int>()();
        TestFunctor<unsigned int>()();
        TestFunctor<long>()();
        TestFunctor<unsigned long>()();
        TestFunctor<long long>()();
        TestFunctor<unsigned long long>()();
        TestFunctor<wchar_t>();
#ifndef _LIBCPP_HAS_NO_UNICODE_CHARS
        TestFunctor<char16_t>()();
        TestFunctor<char32_t>()();
#endif
        TestFunctor<  int8_t>()();
        TestFunctor< uint8_t>()();
        TestFunctor< int16_t>()();
        TestFunctor<uint16_t>()();
        TestFunctor< int32_t>()();
        TestFunctor<uint32_t>()();
        TestFunctor< int64_t>()();
        TestFunctor<uint64_t>()();
    }
};

template < template <class TestArg> class TestFunctor >
struct TestEachAtomicType {
    void operator()() const {
        TestEachIntegralType<TestFunctor>()();
        TestFunctor<UserAtomicType>()();
        TestFunctor<PaddedUserAtomicType>()();
#ifndef __APPLE__
        /*
            These aren't going to be lock-free,
            so some libatomic.a is necessary.
        */
        //TestFunctor<WeirdUserAtomicType>()(); //< Actually, nobody is ready for this until P0528
        TestFunctor<LargeUserAtomicType>()();
#endif
        TestFunctor<int*>()();
        TestFunctor<const int*>()();
        TestFunctor<float>()();
        TestFunctor<double>()();
    }
};


#endif // ATOMIC_HELPER_H
