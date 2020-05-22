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

#if TEST_STD_VER >= 11

#include <cstddef>
#include <functional>

class MoveOnly
{
    MoveOnly(const MoveOnly&);
    MoveOnly& operator=(const MoveOnly&);

    int data_;
public:
    constexpr MoveOnly(int data = 1) : data_(data) {}
    constexpr MoveOnly(MoveOnly&& x)
        : data_(x.data_) {x.data_ = 0;}
    constexpr MoveOnly& operator=(MoveOnly&& x)
        {data_ = x.data_; x.data_ = 0; return *this;}

    constexpr int get() const {return data_;}

    constexpr bool operator==(const MoveOnly& x) const {return data_ == x.data_;}
    constexpr bool operator< (const MoveOnly& x) const {return data_ <  x.data_;}
    constexpr MoveOnly operator+(const MoveOnly& x) const { return MoveOnly{data_ + x.data_}; }
    constexpr MoveOnly operator*(const MoveOnly& x) const { return MoveOnly{data_ * x.data_}; }
};

namespace std {

template <>
struct hash<MoveOnly>
{
    typedef MoveOnly argument_type;
    typedef size_t result_type;
    constexpr std::size_t operator()(const MoveOnly& x) const {return x.get();}
};

}

#endif  // TEST_STD_VER >= 11

#endif  // MOVEONLY_H
