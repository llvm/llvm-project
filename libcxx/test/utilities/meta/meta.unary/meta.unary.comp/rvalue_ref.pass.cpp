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
    static_assert(std::is_reference<T>::value, "");
    static_assert(!std::is_arithmetic<T>::value, "");
    static_assert(!std::is_fundamental<T>::value, "");
    static_assert(!std::is_object<T>::value, "");
    static_assert(!std::is_scalar<T>::value, "");
    static_assert( std::is_compound<T>::value, "");
    static_assert(!std::is_member_pointer<T>::value, "");
}

int main()
{
#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES
    test_rvalue_ref<int&&>();
    test_rvalue_ref<const int&&>();
#endif  // _LIBCPP_HAS_NO_RVALUE_REFERENCES
}
