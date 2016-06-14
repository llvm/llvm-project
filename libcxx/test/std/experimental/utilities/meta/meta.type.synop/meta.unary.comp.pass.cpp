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

struct class_type {};

int main()
{
    {
        typedef int & T;
        static_assert(ex::is_reference_v<T>, "");
        static_assert(std::is_same<decltype(ex::is_reference_v<T>), const bool>::value, "");
        static_assert(ex::is_reference_v<T> == std::is_reference<T>::value, "");
    }
    {
        typedef int T;
        static_assert(!ex::is_reference_v<T>, "");
        static_assert(ex::is_reference_v<T> == std::is_reference<T>::value, "");
    }
    {
        typedef int T;
        static_assert(ex::is_arithmetic_v<T>, "");
        static_assert(std::is_same<decltype(ex::is_arithmetic_v<T>), const bool>::value, "");
        static_assert(ex::is_arithmetic_v<T> == std::is_arithmetic<T>::value, "");
    }
    {
        typedef void* T;
        static_assert(!ex::is_arithmetic_v<T>, "");
        static_assert(ex::is_arithmetic_v<T> == std::is_arithmetic<T>::value, "");
    }
    {
        typedef int T;
        static_assert(ex::is_fundamental_v<T>, "");
        static_assert(std::is_same<decltype(ex::is_fundamental_v<T>), const bool>::value, "");
        static_assert(ex::is_fundamental_v<T> == std::is_fundamental<T>::value, "");
    }
    {
        typedef class_type T;
        static_assert(!ex::is_fundamental_v<T>, "");
        static_assert(ex::is_fundamental_v<T> == std::is_fundamental<T>::value, "");
    }
    {
        typedef class_type T;
        static_assert(ex::is_object_v<T>, "");
        static_assert(std::is_same<decltype(ex::is_object_v<T>), const bool>::value, "");
        static_assert(ex::is_object_v<T> == std::is_object<T>::value, "");
    }
    {
        typedef void T;
        static_assert(!ex::is_object_v<T>, "");
        static_assert(ex::is_object_v<T> == std::is_object<T>::value, "");
    }
    {
        typedef int T;
        static_assert(ex::is_scalar_v<T>, "");
        static_assert(std::is_same<decltype(ex::is_scalar_v<T>), const bool>::value, "");
        static_assert(ex::is_scalar_v<T> == std::is_scalar<T>::value, "");
    }
    {
        typedef void T;
        static_assert(!ex::is_scalar_v<T>, "");
        static_assert(ex::is_scalar_v<T> == std::is_scalar<T>::value, "");
    }
    {
        typedef void* T;
        static_assert(ex::is_compound_v<T>, "");
        static_assert(std::is_same<decltype(ex::is_compound_v<T>), const bool>::value, "");
        static_assert(ex::is_compound_v<T> == std::is_compound<T>::value, "");
    }
    {
        typedef void T;
        static_assert(!ex::is_compound_v<T>, "");
        static_assert(ex::is_compound_v<T> == std::is_compound<T>::value, "");
    }
    {
        typedef int class_type::*T;
        static_assert(ex::is_member_pointer_v<T>, "");
        static_assert(std::is_same<decltype(ex::is_member_pointer_v<T>), const bool>::value, "");
        static_assert(ex::is_member_pointer_v<T> == std::is_member_pointer<T>::value, "");
    }
    {
        typedef int T;
        static_assert(!ex::is_member_pointer_v<T>, "");
        static_assert(ex::is_member_pointer_v<T> == std::is_member_pointer<T>::value, "");
    }
}

