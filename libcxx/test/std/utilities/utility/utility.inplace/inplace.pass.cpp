//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <utility>

// struct in_place_t {
//   explicit in_place_t() = default;
// };
// inline constexpr in_place_t in_place{};

// template <class T>
//   struct in_place_type_t {
//     explicit in_place_type_t() = default;
//   };
// template <class T>
//   inline constexpr in_place_type_t<T> in_place_type{};

// template <size_t I>
//   struct in_place_index_t {
//     explicit in_place_index_t() = default;
//   };
// template <size_t I>
//   inline constexpr in_place_index_t<I> in_place_index{};

#include <utility>
#include <cassert>
#include <memory>

#include "test_macros.h"

template <class Tp, class Up>
constexpr bool check_tag(Up) {
    return std::is_same<Tp, std::decay_t<Tp>>::value
        && std::is_same<Tp, Up>::value;
}

int main(int, char**) {
    // test in_place_t
    {
        using T = std::in_place_t;
        static_assert(check_tag<T>(std::in_place));
    }
    // test in_place_type_t
    {
        using T1 = std::in_place_type_t<void>;
        using T2 = std::in_place_type_t<int>;
        using T3 = std::in_place_type_t<const int>;
        static_assert(!std::is_same<T1, T2>::value && !std::is_same<T1, T3>::value);
        static_assert(!std::is_same<T2, T3>::value);
        static_assert(check_tag<T1>(std::in_place_type<void>));
        static_assert(check_tag<T2>(std::in_place_type<int>));
        static_assert(check_tag<T3>(std::in_place_type<const int>));
    }
    // test in_place_index_t
    {
        using T1 = std::in_place_index_t<0>;
        using T2 = std::in_place_index_t<1>;
        using T3 = std::in_place_index_t<static_cast<size_t>(-1)>;
        static_assert(!std::is_same<T1, T2>::value && !std::is_same<T1, T3>::value);
        static_assert(!std::is_same<T2, T3>::value);
        static_assert(check_tag<T1>(std::in_place_index<0>));
        static_assert(check_tag<T2>(std::in_place_index<1>));
        static_assert(check_tag<T3>(std::in_place_index<static_cast<size_t>(-1)>));
    }

  return 0;
}
