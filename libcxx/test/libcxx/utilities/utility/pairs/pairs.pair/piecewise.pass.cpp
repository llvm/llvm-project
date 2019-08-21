//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <utility>

// template <class T1, class T2> struct pair

// template <class... Args1, class... Args2>
//     pair(piecewise_construct_t, tuple<Args1...> first_args,
//                                 tuple<Args2...> second_args);

#include <tuple>
#include <type_traits>
#include <utility>

#include "archetypes.h"

#include "test_macros.h"


int main(int, char**) {
    using NonThrowingConvert = NonThrowingTypes::ConvertingType;
    using ThrowingConvert = NonTrivialTypes::ConvertingType;
    static_assert(!std::is_nothrow_constructible<std::pair<ThrowingConvert, ThrowingConvert>,
                                                 std::piecewise_construct_t, std::tuple<int, int>, std::tuple<long, long>>::value, "");
    static_assert(!std::is_nothrow_constructible<std::pair<NonThrowingConvert, ThrowingConvert>,
                                                 std::piecewise_construct_t, std::tuple<int, int>, std::tuple<long, long>>::value, "");
    static_assert(!std::is_nothrow_constructible<std::pair<ThrowingConvert, NonThrowingConvert>,
                                                 std::piecewise_construct_t, std::tuple<int, int>, std::tuple<long, long>>::value, "");
    static_assert( std::is_nothrow_constructible<std::pair<NonThrowingConvert, NonThrowingConvert>,
                                                 std::piecewise_construct_t, std::tuple<int, int>, std::tuple<long, long>>::value, "");

  return 0;
}
