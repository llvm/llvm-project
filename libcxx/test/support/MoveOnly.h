//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MOVEONLY_H
#define MOVEONLY_H

#include "test_macros.h"

#include <cstddef>
#include <functional>

class MoveOnly
{
    int data_;
public:
    TEST_CONSTEXPR MoveOnly(int data = 1) : data_(data) {}

    MoveOnly(const MoveOnly&) = delete;
    MoveOnly& operator=(const MoveOnly&) = delete;

    TEST_CONSTEXPR_CXX14 MoveOnly(MoveOnly&& x) TEST_NOEXCEPT
        : data_(x.data_) {x.data_ = 0;}
    TEST_CONSTEXPR_CXX14 MoveOnly& operator=(MoveOnly&& x)
        {data_ = x.data_; x.data_ = 0; return *this;}

    TEST_CONSTEXPR int get() const {return data_;}

    friend TEST_CONSTEXPR bool operator==(const MoveOnly& x, const MoveOnly& y)
        { return x.data_ == y.data_; }
    friend TEST_CONSTEXPR bool operator!=(const MoveOnly& x, const MoveOnly& y)
        { return x.data_ != y.data_; }
    friend TEST_CONSTEXPR bool operator< (const MoveOnly& x, const MoveOnly& y)
        { return x.data_ <  y.data_; }
    friend TEST_CONSTEXPR bool operator<=(const MoveOnly& x, const MoveOnly& y)
        { return x.data_ <= y.data_; }
    friend TEST_CONSTEXPR bool operator> (const MoveOnly& x, const MoveOnly& y)
        { return x.data_ >  y.data_; }
    friend TEST_CONSTEXPR bool operator>=(const MoveOnly& x, const MoveOnly& y)
        { return x.data_ >= y.data_; }

#if TEST_STD_VER > 17
    friend constexpr auto operator<=>(const MoveOnly&, const MoveOnly&) = default;
#endif // TEST_STD_VER > 17

    TEST_CONSTEXPR_CXX14 MoveOnly operator+(const MoveOnly& x) const
        { return MoveOnly(data_ + x.data_); }
    TEST_CONSTEXPR_CXX14 MoveOnly operator*(const MoveOnly& x) const
        { return MoveOnly(data_ * x.data_); }

    template<class T, class U>
    friend void operator,(T t, U u) = delete;
};


template <>
struct std::hash<MoveOnly>
{
    typedef MoveOnly argument_type;
    typedef size_t result_type;
    TEST_CONSTEXPR size_t operator()(const MoveOnly& x) const {return static_cast<size_t>(x.get());}
};

#endif // MOVEONLY_H
