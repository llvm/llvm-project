//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// type_traits

// rvalue_ref

#include <type_traits>

template <class T>
void test_rvalue_ref()
{
    static_assert(!std::is_void<T>::value, "");
    static_assert(!std::is_integral<T>::value, "");
    static_assert(!std::is_floating_point<T>::value, "");
    static_assert(!std::is_array<T>::value, "");
    static_assert(!std::is_pointer<T>::value, "");
    static_assert(!std::is_lvalue_reference<T>::value, "");
    static_assert( std::is_rvalue_reference<T>::value, "");
    static_assert(!std::is_member_object_pointer<T>::value, "");
    static_assert(!std::is_member_function_pointer<T>::value, "");
    static_assert(!std::is_enum<T>::value, "");
    static_assert(!std::is_union<T>::value, "");
    static_assert(!std::is_class<T>::value, "");
    static_assert(!std::is_function<T>::value, "");
}

int main()
{
#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES
    test_rvalue_ref<int&&>();
    test_rvalue_ref<const int&&>();
#endif  // _LIBCPP_HAS_NO_RVALUE_REFERENCES
}
