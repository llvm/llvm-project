//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <array>

// template <size_t I, class T, size_t N> T&& get(array<T, N>&& a);

// UNSUPPORTED: c++03

#include <array>
#include <memory>
#include <utility>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
        typedef std::unique_ptr<double> T;
        typedef std::array<T, 1> C;
        C c = {std::unique_ptr<double>(new double(3.5))};
        T t = std::get<0>(std::move(c));
        assert(*t == 3.5);
    }

    return 0;
}
