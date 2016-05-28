//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11
// <experimental/type_traits>

#include <experimental/type_traits>

namespace ex = std::experimental;

int main()
{
    {
        typedef char T;
        static_assert(ex::alignment_of_v<T> == 1, "");
        static_assert(std::is_same<decltype(ex::alignment_of_v<T>), const std::size_t>::value, "");
        static_assert(ex::alignment_of_v<T> == std::alignment_of<T>::value, "");
    }
    {
        typedef char(T)[1][1][1];
        static_assert(ex::rank_v<T> == 3, "");
        static_assert(std::is_same<decltype(ex::rank_v<T>), const std::size_t>::value, "");
        static_assert(ex::rank_v<T> == std::rank<T>::value, "");
    }
    {
        typedef void T;
        static_assert(ex::rank_v<T> == 0, "");
        static_assert(ex::rank_v<T> == std::rank<T>::value, "");
    }
    {
        typedef char(T)[2][3][4];
        static_assert(ex::extent_v<T> == 2, "");
        static_assert(std::is_same<decltype(ex::extent_v<T>), const std::size_t>::value, "");
        static_assert(ex::extent_v<T> == std::extent<T>::value, "");
    }
    {
        typedef char(T)[2][3][4];
        static_assert(ex::extent_v<T, 0> == 2, "");
        static_assert(ex::extent_v<T, 0> == std::extent<T, 0>::value, "");
    }
    {
        typedef char(T)[2][3][4];
        static_assert(ex::extent_v<T, 1> == 3, "");
        static_assert(ex::extent_v<T, 1> == std::extent<T, 1>::value, "");
    }
    {
        typedef char(T)[2][3][4];
        static_assert(ex::extent_v<T, 5> == 0, "");
        static_assert(ex::extent_v<T, 5> == std::extent<T, 5>::value, "");
    }
    {
        typedef void T;
        static_assert(ex::extent_v<T, 0> == 0, "");
        static_assert(ex::extent_v<T, 0> == std::extent<T, 0>::value, "");
    }
}
