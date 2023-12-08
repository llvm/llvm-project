//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// <optional>

// constexpr const T& optional<T>::value() const &;

#include <optional>
#include <type_traits>
#include <cassert>

using std::optional;

struct X
{
    constexpr int test() const {return 3;}
    int test() {return 4;}
};

int main(int, char**)
{
    {
        constexpr optional<X> opt;
        static_assert(opt.value().test() == 3, "");
    }

  return 0;
}
