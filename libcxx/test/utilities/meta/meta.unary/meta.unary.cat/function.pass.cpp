//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// type_traits

// function

#include <type_traits>

template <class T>
void test_function_imp()
{
    static_assert(!std::is_void<T>::value, "");
    static_assert(!std::is_integral<T>::value, "");
    static_assert(!std::is_floating_point<T>::value, "");
    static_assert(!std::is_array<T>::value, "");
    static_assert(!std::is_pointer<T>::value, "");
    static_assert(!std::is_lvalue_reference<T>::value, "");
    static_assert(!std::is_rvalue_reference<T>::value, "");
    static_assert(!std::is_member_object_pointer<T>::value, "");
    static_assert(!std::is_member_function_pointer<T>::value, "");
    static_assert(!std::is_enum<T>::value, "");
    static_assert(!std::is_union<T>::value, "");
    static_assert(!std::is_class<T>::value, "");
    static_assert( std::is_function<T>::value, "");
}

template <class T>
void test_function()
{
    test_function_imp<T>();
    test_function_imp<const T>();
    test_function_imp<volatile T>();
    test_function_imp<const volatile T>();
}

int main()
{
    test_function<void ()>();
    test_function<void (int)>();
    test_function<int (double)>();
    test_function<int (double, char)>();
}
