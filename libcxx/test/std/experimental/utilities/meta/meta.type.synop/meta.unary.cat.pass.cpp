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
enum enum_type {};
union union_type {};

int main()
{
    {
        typedef void T;
        static_assert(ex::is_void_v<T>, "");
        static_assert(std::is_same<decltype(ex::is_void_v<T>), const bool>::value, "");
        static_assert(ex::is_void_v<T> == std::is_void<T>::value, "");
    }
    {
        typedef int T;
        static_assert(!ex::is_void_v<T>, "");
        static_assert(ex::is_void_v<T> == std::is_void<T>::value, "");
    }
    {
        typedef decltype(nullptr) T;
        static_assert(ex::is_null_pointer_v<T>, "");
        static_assert(std::is_same<decltype(ex::is_null_pointer_v<T>), const bool>::value, "");
        static_assert(ex::is_null_pointer_v<T> == std::is_null_pointer<T>::value, "");
    }
    {
        typedef int T;
        static_assert(!ex::is_null_pointer_v<T>, "");
        static_assert(ex::is_null_pointer_v<T> == std::is_null_pointer<T>::value, "");
    }
    {
        typedef int T;
        static_assert(ex::is_integral_v<T>, "");
        static_assert(std::is_same<decltype(ex::is_integral_v<T>), const bool>::value, "");
        static_assert(ex::is_integral_v<T> == std::is_integral<T>::value, "");
    }
    {
        typedef void T;
        static_assert(!ex::is_integral_v<T>, "");
        static_assert(ex::is_integral_v<T> == std::is_integral<T>::value, "");
    }
    {
        typedef float T;
        static_assert(ex::is_floating_point_v<T>, "");
        static_assert(std::is_same<decltype(ex::is_floating_point_v<T>), const bool>::value, "");
        static_assert(ex::is_floating_point_v<T> == std::is_floating_point<T>::value, "");
    }
    {
        typedef int T;
        static_assert(!ex::is_floating_point_v<T>, "");
        static_assert(ex::is_floating_point_v<T> == std::is_floating_point<T>::value, "");
    }
    {
        typedef int(T)[42];
        static_assert(ex::is_array_v<T>, "");
        static_assert(std::is_same<decltype(ex::is_array_v<T>), const bool>::value, "");
        static_assert(ex::is_array_v<T> == std::is_array<T>::value, "");
    }
    {
        typedef int T;
        static_assert(!ex::is_array_v<T>, "");
        static_assert(ex::is_array_v<T> == std::is_array<T>::value, "");
    }
    {
        typedef void* T;
        static_assert(ex::is_pointer_v<T>, "");
        static_assert(std::is_same<decltype(ex::is_pointer_v<T>), const bool>::value, "");
        static_assert(ex::is_pointer_v<T> == std::is_pointer<T>::value, "");
    }
    {
        typedef int T;
        static_assert(!ex::is_pointer_v<T>, "");
        static_assert(ex::is_pointer_v<T> == std::is_pointer<T>::value, "");
    }
    {
        typedef int & T;
        static_assert(ex::is_lvalue_reference_v<T>, "");
        static_assert(std::is_same<decltype(ex::is_lvalue_reference_v<T>), const bool>::value, "");
        static_assert(ex::is_lvalue_reference_v<T> == std::is_lvalue_reference<T>::value, "");
    }
    {
        typedef int T;
        static_assert(!ex::is_lvalue_reference_v<T>, "");
        static_assert(ex::is_lvalue_reference_v<T> == std::is_lvalue_reference<T>::value, "");
    }
    {
        typedef int && T;
        static_assert(ex::is_rvalue_reference_v<T>, "");
        static_assert(std::is_same<decltype(ex::is_rvalue_reference_v<T>), const bool>::value, "");
        static_assert(ex::is_rvalue_reference_v<T> == std::is_rvalue_reference<T>::value, "");
    }
    {
        typedef int T;
        static_assert(!ex::is_rvalue_reference_v<T>, "");
        static_assert(ex::is_rvalue_reference_v<T> == std::is_rvalue_reference<T>::value, "");
    }
    {
        typedef int class_type::*T;
        static_assert(ex::is_member_object_pointer_v<T>, "");
        static_assert(std::is_same<decltype(ex::is_member_object_pointer_v<T>), const bool>::value, "");
        static_assert(ex::is_member_object_pointer_v<T> == std::is_member_object_pointer<T>::value, "");
    }
    {
        typedef int T;
        static_assert(!ex::is_member_object_pointer_v<T>, "");
        static_assert(ex::is_member_object_pointer_v<T> == std::is_member_object_pointer<T>::value, "");
    }
    {
        typedef void(class_type::*T)();
        static_assert(ex::is_member_function_pointer_v<T>, "");
        static_assert(std::is_same<decltype(ex::is_member_function_pointer_v<T>), const bool>::value, "");
        static_assert(ex::is_member_function_pointer_v<T> == std::is_member_function_pointer<T>::value, "");
    }
    {
        typedef int T;
        static_assert(!ex::is_member_function_pointer_v<T>, "");
        static_assert(ex::is_member_function_pointer_v<T> == std::is_member_function_pointer<T>::value, "");
    }
    {
        typedef enum_type T;
        static_assert(ex::is_enum_v<T>, "");
        static_assert(std::is_same<decltype(ex::is_enum_v<T>), const bool>::value, "");
        static_assert(ex::is_enum_v<T> == std::is_enum<T>::value, "");
    }
    {
        typedef int T;
        static_assert(!ex::is_enum_v<T>, "");
        static_assert(ex::is_enum_v<T> == std::is_enum<T>::value, "");
    }
    {
        typedef union_type T;
        static_assert(ex::is_union_v<T>, "");
        static_assert(std::is_same<decltype(ex::is_union_v<T>), const bool>::value, "");
        static_assert(ex::is_union_v<T> == std::is_union<T>::value, "");
    }
    {
        typedef int T;
        static_assert(!ex::is_union_v<T>, "");
        static_assert(ex::is_union_v<T> == std::is_union<T>::value, "");
    }
    {
        typedef class_type T;
        static_assert(ex::is_class_v<T>, "");
        static_assert(std::is_same<decltype(ex::is_class_v<T>), const bool>::value, "");
        static_assert(ex::is_class_v<T> == std::is_class<T>::value, "");
    }
    {
        typedef int T;
        static_assert(!ex::is_class_v<T>, "");
        static_assert(ex::is_class_v<T> == std::is_class<T>::value, "");
    }
    {
        typedef void(T)();
        static_assert(ex::is_function_v<T>, "");
        static_assert(std::is_same<decltype(ex::is_function_v<T>), const bool>::value, "");
        static_assert(ex::is_function_v<T> == std::is_function<T>::value, "");
    }
    {
        typedef int T;
        static_assert(!ex::is_function_v<T>, "");
        static_assert(ex::is_function_v<T> == std::is_function<T>::value, "");
    }
}

