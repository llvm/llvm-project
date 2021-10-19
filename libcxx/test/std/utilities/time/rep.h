//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef REP_H
#define REP_H

#include "test_macros.h"

class Rep
{
    int data_;
public:
    TEST_CONSTEXPR Rep() : data_(-1) {}
    explicit TEST_CONSTEXPR Rep(int i) : data_(i) {}

    bool TEST_CONSTEXPR operator==(int i) const {return data_ == i;}
    bool TEST_CONSTEXPR operator==(const Rep& r) const {return data_ == r.data_;}

    Rep& operator*=(Rep x) {data_ *= x.data_; return *this;}
    Rep& operator/=(Rep x) {data_ /= x.data_; return *this;}
};

// This is PR#41130

struct NotARep {};

// std::chrono:::duration has only '*', '/' and '%' taking a "Rep" parameter

// Multiplication is commutative, division is not.
template <class Rep, class Period>
std::chrono::duration<Rep, Period>
operator*(std::chrono::duration<Rep, Period> d, NotARep) { return d; }

template <class Rep, class Period>
std::chrono::duration<Rep, Period>
operator*(NotARep, std::chrono::duration<Rep, Period> d) { return d; }

template <class Rep, class Period>
std::chrono::duration<Rep, Period>
operator/(std::chrono::duration<Rep, Period> d, NotARep) { return d; }

template <class Rep, class Period>
std::chrono::duration<Rep, Period>
operator%(std::chrono::duration<Rep, Period> d, NotARep) { return d; }

// op= is not commutative.
template <class Rep, class Period>
std::chrono::duration<Rep, Period>&
operator*=(std::chrono::duration<Rep, Period>& d, NotARep) { return d; }

template <class Rep, class Period>
std::chrono::duration<Rep, Period>&
operator/=(std::chrono::duration<Rep, Period>& d, NotARep) { return d; }

template <class Rep, class Period>
std::chrono::duration<Rep, Period>&
operator%=(std::chrono::duration<Rep, Period>& d, NotARep) { return d; }

#endif // REP_H
