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

struct base_type {};
struct derived_type : base_type {};

int main()
{
    {
        typedef int T;
        typedef int U;
        static_assert(ex::is_same_v<T, U>, "");
        static_assert(std::is_same<decltype(ex::is_same_v<T, U>), const bool>::value, "");
        static_assert(ex::is_same_v<T, U> == std::is_same<T, U>::value, "");
    }
    {
        typedef int T;
        typedef long U;
        static_assert(!ex::is_same_v<T, U>, "");
        static_assert(ex::is_same_v<T, U> == std::is_same<T, U>::value, "");
    }
    {
        typedef base_type T;
        typedef derived_type U;
        static_assert(ex::is_base_of_v<T, U>, "");
        static_assert(std::is_same<decltype(ex::is_base_of_v<T, U>), const bool>::value, "");
        static_assert(ex::is_base_of_v<T, U> == std::is_base_of<T, U>::value, "");
    }
    {
        typedef int T;
        typedef int U;
        static_assert(!ex::is_base_of_v<T, U>, "");
        static_assert(ex::is_base_of_v<T, U> == std::is_base_of<T, U>::value, "");
    }
    {
        typedef int T;
        typedef long U;
        static_assert(ex::is_convertible_v<T, U>, "");
        static_assert(std::is_same<decltype(ex::is_convertible_v<T, U>), const bool>::value, "");
        static_assert(ex::is_convertible_v<T, U> == std::is_convertible<T, U>::value, "");
    }
    {
        typedef void T;
        typedef int U;
        static_assert(!ex::is_convertible_v<T, U>, "");
        static_assert(ex::is_convertible_v<T, U> == std::is_convertible<T, U>::value, "");
    }
}

